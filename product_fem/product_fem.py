import numpy as np
import scipy.sparse as sps
from fenics import *

## TODO
### write tests
def default_product_boundary(W, x, y):
    # rule which determines if (x,y) is on the product boundary
    # default product boundary is bdy(M)xM cup Mxbdy(M)
    marginal_bc = DirichletBC(W.marginal_function_space(), 0, 'on_boundary')
    marginal_bc_dofs = marginal_bc.get_boundary_values().keys()
    
    # loop over marginal boundary dofs i
    # determine if x or y near marginal boundary
    for i_bdy in marginal_bc_dofs:
        # this probably doesn't work with 2d mesh
        x_bdy = near(x, W.dofmap.marginal_dof_coords[i_bdy])
        y_bdy = near(y, W.dofmap.marginal_dof_coords[i_bdy])
        on_bdy = x_bdy or y_bdy
        if on_bdy: break
            
    return on_bdy
            
def form_to_array(form):
    rank = len(form.arguments())
    arr = assemble(form)
    if rank==0:
        array = arr.real
    elif rank==1:
        array = arr[:]
    elif rank==2:
        array = arr.array()
    return array

def assemble_kron(forms):
    # if 2 forms are given, compute kron(B, C) + kron(C, B)
    if len(forms)==2:
        B_form, C_form = forms
        B = form_to_array(B_form)
        C = form_to_array(C_form)
        A = np.kron(B, C) + np.kron(C, B)
    
    # if 4 forms are given, compute kron(B, C) + kron(D, E)
    elif len(forms)==4:
        B_form, C_form, D_form, E_form = forms
        B = form_to_array(B_form)
        C = form_to_array(C_form)
        D = form_to_array(D_form)
        E = form_to_array(E_form)
        A = np.kron(B, C) +  np.kron(D, E)
        
    return A
        
def assemble_product(A_forms, b_forms):
    # assembles linear system AU=b where
    # A = kron(B,C) + kron(C,B)
    # b = kron(c,d)
    assert len(b_forms)==2

    # LHS assembly
    A = assemble_kron(A_forms)

    # RHS assembly
    if not isinstance(b_forms[0], list):
        c_form, d_form = b_forms
        c = form_to_array(c_form)
        d = form_to_array(d_form)
        b = np.kron(c, d)
    else:
        c_forms, d_forms = b_forms
        assert len(c_forms)==len(d_forms)
        b = []
        for i in range(len(c_forms)):
            Xi = form_to_array(c_forms[i])
            Yi = form_to_array(d_forms[i])
            b.append(np.kron(Xi, Yi))
        b = np.sum(b, axis=0)

    return A, b

def assemble_product_system(A_forms, b_forms, bc=None):
    # assemble forms
    A, b = assemble_product(A_forms, b_forms)
    if bc is not None:
        A, b = bc.apply(A, b)
    return A, b
    

class ProductDofMap:
    # main usage is to obtain bijections between
    # dofs ij <-> product dofs (i,j) (defined by Kronecker product)
    # dofs ij <-> product coordinates (x_i, y_j)
    def __init__(self, function_space):
        # marginal dofs and coordinates
        dofmap = function_space.dofmap()
        dofs = dofmap.dofs()
        dof_coords = function_space.tabulate_dof_coordinates()
        
        # product space dofs ij and coordinates (x_i, y_j)
        self.dofs = [ij for ij in range(len(dofs)**2)] # sets stiffness sparsity pattern
        self.product_dofs = [(i,j) for i in dofs for j in dofs] 
        self.product_dof_coords = [(x.item(),y.item()) for x in dof_coords for y in dof_coords]
        
        # dictionaries for dof/coordinate mapping
        self._dofs_to_product_dofs = dict(zip(self.dofs, self.product_dofs)) # ij -> (i,j)
        self._product_dofs_to_dofs = dict(zip(self.product_dofs, self.dofs)) # (i,j) -> ij 
        self._dofs_to_coords = dict(zip(self.dofs, self.product_dof_coords)) # ij -> (x_i, y_j)
        self._product_dofs_to_coords = dict(zip(self.product_dofs, self.product_dof_coords)) # (i,j)->(x_i,y_j)
        
        # save marginal space dofs and coordinates
        self.marginal_function_space = function_space
        self.marginal_dofmap = dofmap
        self.marginal_dofs = dofs 
        self.marginal_dof_coords = dof_coords
        
        
class ProductFunctionSpace:
    def __init__(self, V):
        # V is fenics.FunctionSpace
        self._marginal_function_space = V
        self._marginal_mesh = V.mesh()
        self.dofmap = ProductDofMap(V)
        
    def dofs(self):
        # need to do bijections that can be
        # restricted to the boundary
        # product_dofs <-kron-> marginal_dofs
        return self.dofmap._dofs_to_product_dofs
    
    def tabulate_dof_coordinates(self):
        # marginal_dofs <-dof_coords-> marginal_coords
        # product_dofs <--> marginal_coords 
        # ^^factors through the previous 2 bijections
        return self.dofmap._dofs_to_coords
    
    def marginal_function_space(self):
        return self._marginal_function_space
    
    def marginal_mesh(self):
        return self._marginal_mesh
    

class ProductFunction:
    def __init__(self, product_function_space):
        # initializes product space function 
        # f(x,y) = sum_ij f_ij phi_i(x)phi_j(y)
        # where f_ij = f(x_i, y_j)
        # by default f_ij=0 for all ij
        n_dofs = len(product_function_space.dofmap.dofs)
        self.function = np.zeros(n_dofs)
        
    def assign(self, f):
        # assigns values in array f to product function f(x,y)
        # i.e. f contains f_ij
        self.function = f
        
        
class ProductDirichletBC:
    def __init__(self, W, u_bdy, on_product_boundary='default'):
        # on_product_boundary: (x,y) -> True if (x,y) on product boundary, else False
        # u_bdy is a map (x,y) -> u(x,y) given that on_bound(x,y)==True
        # W is ProductFunctionSpace, can use .dofmap to help with on_bound
        if on_product_boundary in ['on_boundary', 'default']:
            on_product_boundary = default_product_boundary
        if isinstance(u_bdy, (int, float)):
            u_bv = float(u_bdy)
            u_bdy = lambda x,y: u_bv
            
        self.product_function_space = W
        self.marginal_function_space = W._marginal_function_space
        self.boundary_values = u_bdy
        self.on_boundary = on_product_boundary
            
    def get_marginal_boundary_dofs(self):
        bc = DirichletBC(self.marginal_function_space, 0, 'on_boundary')
        return bc.get_boundary_values().keys()

    def get_product_boundary_dofs(self):
        # dofs ij where either i or j in marginal bdy dofs
        marginal_bdy_dofs = self.get_marginal_boundary_dofs()
        dofs = self.product_function_space.dofmap._dofs_to_product_dofs # ij->(i,j)
        product_bdy_dofs = []
        for ij, ij_pair in dofs.items():
            i, j = ij_pair
            if i in marginal_bdy_dofs or j in marginal_bdy_dofs:
                product_bdy_dofs.append(ij)
        return product_bdy_dofs

    def get_product_boundary_coords(self):
        prod_bdy_dofs = self.get_product_boundary_dofs()
        dof_to_coords = self.product_function_space.dofmap._dofs_to_coords # ij->(x_i,y_j)
        product_bdy_coords = [dof_to_coords[ij] for ij in prod_bdy_dofs]
        return product_bdy_coords
            
    def apply(self, A, b):
        # applies desired bdy conds to system AU=b
        # for bdy dof ij, replace A[ij] with e[ij]
        # replace b[ij] with u_bdy(x_i, y_j)
        e = np.eye(len(A))
        prod_bdy_dofs = self.get_product_boundary_dofs() # ij on boundary
        prod_bdy_coords = self.get_product_boundary_coords() # (x_i, y_j) on boundary
        bvs = [self.boundary_values(xy[0], xy[1]) for xy in prod_bdy_coords]
        for k, ij in enumerate(prod_bdy_dofs):
            A[ij] = e[ij]
            b[ij] = bvs[k]
        return A, b
    
    