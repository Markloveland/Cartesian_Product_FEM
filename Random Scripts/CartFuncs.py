from __future__ import print_function
from fenics import *
from ufl import nabla_div
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import petsc4py
petsc4py.init()
from petsc4py import PETSc

#all functions necessary to run Test Case 4
#this expression requires sigma to be in second index
def Gauss_Expression_1D(F_peak,F_std,HS):
    aux1 = aux1 = HS**2/(16*np.sqrt(2*np.pi)*F_std)
    aux3 = 2*F_std**2
    #CTOT = np.sqrt(0.5*500/np.pi)/(1-0.25/500)
    E=Expression('aux1*exp(-pow(x[1]-2*pi*F_peak,2)/aux3)',
                degree=4,aux1=aux1,aux3=aux3,F_peak=F_peak)
    return E

def Gauss_Expression_IC(F_peak,F_std,HS):
    aux1 = aux1 = HS**2/(16*np.sqrt(2*np.pi)*F_std)
    aux3 = 2*F_std**2
    tol=1e-14
    #CTOT = np.sqrt(0.5*500/np.pi)/(1-0.25/500)
    E=Expression('x[0] < tol ? aux1*exp(-pow(x[1]-2*pi*F_peak,2)/aux3): 0',
                degree=4,aux1=aux1,aux3=aux3,F_peak=F_peak,tol=tol,t=0)
    return E

def Gauss_IC(F_peak,F_std,HS,x,sigmas):
    #takes in dof and paramters
    #returns vector with initial condition values at global DOF
    aux1 = HS**2/(16*np.sqrt(2*np.pi)*F_std)
    aux3 = 2*F_std**2
    tol=1e-14
    E = (x<tol)*aux1*np.exp(-(sigmas - 2*np.pi*F_peak)**2/aux3)
    return E


#an Expression class to evaluate u(x,y) on the 1D domain (range y, x fixed)
#maybe modify so input can determine if we want range y (fixed x) or range x (fixed y)
#this case want space to be x, sigma to be y
class my1DExpression(UserExpression):
    def __init__(self,u_2d,x,**kwargs):
        super().__init__(**kwargs)
        self.u_2d=u_2d
        self.x=x
        #Expression.__init__(self)
        self._vx= np.array([0.])
        self._pt_x= np.array([0.,0.])
    def eval(self, values, x):
        self._pt_x[0]= self.x
        self._pt_x[1]= x[0]
        self.u_2d.eval(self._vx,self._pt_x)
        values[0] = self._vx[0]
        
#need a function to calculate wave speed (phase and group) and wavenumber
def swan_calc_wavespeeds_k(sigmas,depths,g=9.81):
    #taken from Jessica's code
    cg_out=np.zeros(len(sigmas))
    k_out=np.zeros(len(sigmas))
    WGD=np.sqrt(depths/g)*g
    SND=sigmas*np.sqrt(depths/g)
    
    shallow_range=np.argwhere(SND<1e-6)
    mid_range=np.argwhere((SND<2.5)&(SND>=1e-6))
    deep_range=np.argwhere(SND>=2.5)
    
    def cg_mid(SND,g,depths,sigmas):
        SND2=SND*SND
        C=np.sqrt(g*depths/(SND2 +1/(1+0.666*SND2+.445*SND2**2 
                                     - 0.105*SND2**3 + 0.272*SND2**4)))
        KND=sigmas*depths/C
        
        FAC1=2*KND/np.sinh(2*KND)
        N=0.5*(1+FAC1)
        return N*C,sigmas/C
    def cg_deep(g,sigmas):
        return 0.5*g/sigmas
    def cg_shallow(WGD):
        return WGD

    cg_out[shallow_range]=cg_shallow(WGD[shallow_range])
    k_out[shallow_range]=SND[shallow_range]/depths[shallow_range]
    cg_out[mid_range],k_out[mid_range]=cg_mid(SND[mid_range],g,depths[mid_range],sigmas[mid_range])
    cg_out[deep_range]=cg_deep(g,sigmas[deep_range])
    k_out[deep_range]=sigmas[deep_range]**2/g

    return cg_out,k_out

def calc_c_sigma(sigmas,k,depths,c_g,currents,z_coords,loc_num):
    ##Inputs
    #sigmas - takes in a vector of sigma values (sigma coordinate at each d.o.f) at one point in physical spaxe
    #thetas
    #k - wavenumber corresponding to the given sigmas and thetas
    #c_g - corresponding group velocity
    #(note sigmas, thetas,k,c_g should all be same length which is number of d.o.f in one 2D slice)
    #depths - depths in meters of ALL physical coordinates (not just current one)
    #z_coords - unique coordinates in physical space (not just current one)
    #currents - water velocity at ALL physical coordinates
    #loc_num - the number of the physical point (starts at 0)
    #(note depths,z_coords,currents should all be same length which is the number of unique points in physical space)
    ##Outputs
    #outputs a vector same size as sigmas that is the c_sigma at all d.o.f in ONE 2D slice 
    #of the 3D domain corresponding to a single point in physical space for c_theta and c_sigma
    
    #for now assuming H is constant in time but can fix this later
    dHdt=0.0
    dHdy = 0.0
    dudy = 0.0
    dvdx = 0.0 #migh not have to be 0, well see
    dvdy = 0.0
    
    #also going to assume v is zero but maybe can change this later
    v=0.0
    u=0.0#currents[loc_num]
    H=depths#[loc_num]
    #calc gradient of H w.r.t. x
    #this is just forward euler but only works for fixed geometry
    #instead we'll hard code for this case
    dHdx=-1.0/200.0
    dudx=0.0
    '''
    if loc_num == 0:
        dHdx =(depths[loc_num+1]-H)/(z_coords[loc_num+1]-z_coords[loc_num])
        dudx = (currents[loc_num+1]-u)/(z_coords[loc_num+1]-z_coords[loc_num])
    elif loc_num == len(z_coords)-1:
        dHdx = (H-depths[loc_num-1])/(z_coords[loc_num]-z_coords[loc_num-1])
        dudx = (u-currents[loc_num-1])/(z_coords[loc_num]-z_coords[loc_num-1])
    else:
        dHdx = (depths[loc_num+1]-depths[loc_num-1])/(z_coords[loc_num+1]-z_coords[loc_num-1])
        dudx = (currents[loc_num+1]-currents[loc_num-1])/(z_coords[loc_num+1]-z_coords[loc_num-1])
    '''
    
    #now calculate velocity vectors
    c_sigma = k*sigmas/(np.sinh(2*k*H)) *(dHdt + u*dHdx + v*dHdy) - c_g*k*(dudx)
    #c_theta = sigmas/(np.sinh(2*k*H))*(dHdx*np.sin(thetas)- dHdy*np.cos(thetas)) + \
    #    dudx*np.cos(thetas)*np.sin(thetas) - dudy*(np.cos(thetas)**2) + dvdx*(np.sin(thetas)**2) \
    #    -dvdy*np.cos(thetas)*np.sin(thetas)
    return c_sigma #,c_theta

###
###
###
#all functions relating to Cartesian product
def cartesian_product_dofs(array1,array2):
    #gives cartesian product of 2 1d vectors
    dim1=len(array1)
    dim2=len(array2)
    out_arr=np.zeros((dim1*dim2,2))
    c=0
    for a in range(dim1):
        for b in range(dim2):
            out_arr[c,:]=[array1[a],array2[b]]
            c=c+1
    return out_arr

def cartesian_product_coords(array1,array2):
    #gives cartesian product of 2 vectors
    dim1,n1=array1.shape
    dim2,n2=array2.shape
    out_arr=np.zeros((dim1*dim2,n1+n2))
    c=0
    for a in range(dim1):
        for b in range(dim2):
            #print(array1[a,:])
            #print(array2[b,:])
            out_arr[c,:]=np.append(array1[a,:],array2[b,:])
            c=c+1
    return out_arr

def cartesian_form_to_kroneck_form(indeces, len_dim_2):
    #designed to take list of indexes in the cartesian form
    #translate to single form as kron product would
    #only for cartesian product between 2 spaces
    #expects matrix with n rows, 2 columns
    num_indeces=indeces.shape[0]
    out_arr=np.zeros(num_indeces)
    for n in range(num_indeces):
        out_arr[n] = indeces[n,0]*len_dim_2 + indeces[n,1]
    return out_arr

def spec_mult(c,K):
    len_dof=len(c)
    for i in range(len_dof):
        for j in range(len_dof):
            K[i,j] = (c[i]+c[j])/2*K[i,j]
    return K

def u_pointwise(x,y):
    #exact solution at each point
    #case 1
    u = np.exp(-x**2-y**2)
    #case 2 homogeneous boundaries
    #u = np.sin(x)*np.sin(y)
    return u
def S_pointwise(x,y):
    #exact S at each point
    #case 1
    S = np.exp(-(x**2 + y**2)) - 2*(x+y)*np.exp(-(x**2 + y**2))
    #case 2 homogeneous boundaries
    #S = np.sin(x)*np.sin(y) + np.cos(x)*np.sin(y) + np.sin(x)*np.cos(y)
    return S
def row_to_row_ind(rows):
    #designed to transform output in csr to a vector with row number for each value
    row_ind = np.zeros(rows[-1],dtype=np.int32)
    for i in range(len(rows)-1):
        row_ind[rows[i]:rows[i+1]]=i
    return row_ind
def get_nnz_vec(rows):
    #takes rows output from csr and gives nnz at each row
    nnz = rows[1:]-rows[:-1]
    return nnz

def kron_PETSC(K1,K2):
    #takes in to AIJ sparse matrices and gives sparse KRON of K1,K2
    # WARNING: assumes square matrices so be careful!!!
    K1_size = K1.getSize()[0]
    K2_size = K2.getSize()[0]
     
    # CSR arrays from K2 matrix
    rows_2,cols_2,vals_2 = K2.getValuesCSR()
    
    # CSR arrrays in K1 matrix
    rows_1,cols_1,vals_1 = K1.getValuesCSR()
    
    
    #from this info formulate nnz vector
    nnz1 = get_nnz_vec(rows_1)
    nnz2 = get_nnz_vec(rows_2)
    nnz = np.kron(nnz1,nnz2)
     

    #establish large matrix to output
    Big_K = PETSc.Mat().create()
    Big_K.setSizes(K1_size*K2_size,K1_size*K2_size)
    Big_K.setType('aij')
    Big_K.setPreallocationNNZ(nnz)
    Big_K.setUp()
    
    row_num = 0 
    for i in range(K1_size):
        #extract first ith row of first sparse matrix
        a_cols = cols_1[rows_1[i]:rows_1[i+1]]
        a_vals = vals_1[rows_1[i]:rows_1[i+1]]
        for j in range(K2_size):
            #extract jth row of second sparse matrix
            b_cols = cols_2[rows_2[j]:rows_2[j+1]]
            b_vals = vals_2[rows_2[j]:rows_2[j+1]]
            #now compute global values and column numbers
            vals = np.kron(a_vals,b_vals)
            cols = np.zeros(len(vals),dtype=np.int32)
            ctr = 0
            num_b_cols=len(b_cols)
            for acol in a_cols:
                offset=acol*K2_size
                cols[ctr*num_b_cols:(ctr+1)*num_b_cols]=b_cols+offset
                ctr=ctr+1
            Big_K.setValues(row_num,cols,vals)
            row_num=row_num+1

    #Big_K.assemble()
    return Big_K

def Mass_assemble_PETSC(K11,K12,K14,K21,K22,K24,boundary_dofs,nnz=0):
    #specific to this problem
    # all matrices must be same size
    #np.kron(K11,K21)-np.kron(K12,K21)-np.kron(K11,K22) + \
    #+np.kron(K14,K21) + \
    #+ np.kron(K11,K24)
    
    #the above assembly gives
    #np.kron(K11,K21-K22+K24) - np.kron(K12-K14,K21)
    # we will relabel kron(K1,K2) + kron(K3,K4)
    #first, get global sparsity pattern    
    #assumes all matrices are square and same size within respective subdomain
    K1_size = K11.getSize()[0]
    K2_size = K21.getSize()[0]
    N_dof = K1_size*K2_size
    K1 = K11
    K2 = K21-K22+K24
    K3 = -K12+K24
    K4 = K21

    #get sparsity pattern for all smaller matrices
    rows_1,cols_1,vals_1 = K1.getValuesCSR()
    rows_2,cols_2,vals_2 = K2.getValuesCSR()
    rows_3,cols_3,vals_3 = K3.getValuesCSR()
    rows_4,cols_4,vals_4 = K4.getValuesCSR()
    
    #this loop will get the nnz, may be just as expensive as constructing the matrix but lets see
    #we know that sparsity of one set of kronecker bois is just kron(nnz1,nnz2) but that won't cut it since we're adding 2 guys
    #'''
    nnz=np.zeros(N_dof,dtype=np.int32)
    ctr=0
    bc_ctr=0
    for i in range(K1_size):

        a1_cols = cols_1[rows_1[i]:rows_1[i+1]]
        a2_cols = cols_3[rows_3[i]:rows_3[i+1]]
        for j in range(K2_size):
            b1_cols = cols_2[rows_2[j]:rows_2[j+1]]
            b2_cols = cols_4[rows_4[j]:rows_4[j+1]]
           
            ab1_cols = np.zeros(len(a1_cols)*len(b1_cols),dtype=np.int32)
            num_a1_col = len(a1_cols)
            num_b1_col = len(b1_cols)
            for k in range(num_a1_col):
                offset=a1_cols[k]*K2_size
                ab1_cols[k*num_b1_col:(k+1)*num_b1_col]=b1_cols+offset 
            ab2_cols = np.zeros(len(a2_cols)*len(b2_cols),dtype=np.int32)
            num_a2_col = len(a2_cols)
            num_b2_col = len(b2_cols)
            for k in range(num_a2_col):
                offset=a2_cols[k]*K2_size
                ab2_cols[k*num_b2_col:(k+1)*num_b2_col]=b2_cols+offset
            
            if np.isin(ctr,boundary_dofs[bc_ctr]):
                nnz[ctr]=1
                bc_ctr=bc_ctr+1
            else:
                nnz[ctr]=len(np.unique(np.concatenate((ab1_cols,ab2_cols),0)))    
            ctr=ctr+1
    print(nnz)    
    #initialize global stiffness matrix
    #'''

  
    Big_K = PETSc.Mat().create()
    Big_K.setSizes(K1_size*K2_size,K1_size*K2_size)
    Big_K.setType('aij')
    Big_K.setPreallocationNNZ(nnz)
    Big_K.setUp()
    
    #now in similar way, loop through and construct matrix
    bc_ctr = 0
    ctr=0
    nnz_save=np.zeros(K1_size*K2_size,dtype=np.int32)
    for i in range(K1_size):

        a1_cols = cols_1[rows_1[i]:rows_1[i+1]]
        a2_cols = cols_3[rows_3[i]:rows_3[i+1]]
        a1_vals = vals_1[rows_1[i]:rows_1[i+1]]
        a2_vals = vals_3[rows_3[i]:rows_3[i+1]]
        for j in range(K2_size):
            b1_cols = cols_2[rows_2[j]:rows_2[j+1]]
            b2_cols = cols_4[rows_4[j]:rows_4[j+1]]
            b1_vals = vals_2[rows_2[j]:rows_2[j+1]]
            b2_vals = vals_4[rows_4[j]:rows_4[j+1]]
           
            ab1_cols = np.zeros(len(a1_cols)*len(b1_cols),dtype=np.int32)
            vals_ab1 = np.kron(a1_vals,b1_vals)
            num_a1_col = len(a1_cols)
            num_b1_col = len(b1_cols)
            for k in range(num_a1_col):
                offset=a1_cols[k]*K2_size
                ab1_cols[k*num_b1_col:(k+1)*num_b1_col]=b1_cols+offset

            
            ab2_cols = np.zeros(len(a2_cols)*len(b2_cols),dtype=np.int32)
            vals_ab2 = np.kron(a2_vals,b2_vals)
            num_a2_col = len(a2_cols)
            num_b2_col = len(b2_cols)
            for k in range(num_a2_col):
                offset=a2_cols[k]*K2_size
                ab2_cols[k*num_b2_col:(k+1)*num_b2_col]=b2_cols+offset
            
            #create a final vals that has overlapping parts added, non overlapping parts as is
            #first find the intersect and add the two
            #int_locs=np.intersect1d(ab1_cols,ab2_cols)
            
            #probably a more efficient way but for now do:
            cols=np.union1d(ab1_cols,ab2_cols)
            vals = np.zeros(len(cols))

            aa=0
            bb=0
            cc=0
            for colnum in cols:
                #see which one is what
                val1=0
                val2=0
                if np.isin(colnum,ab1_cols):
                    val1=vals_ab1[aa]
                    aa=aa+1 
                if np.isin(colnum,ab1_cols):
                    val2=vals_ab2[bb]
                    bb=bb+1
                vals[cc]=val1+val2
                cc=cc+1
            
            if np.isin(ctr,boundary_dofs):
                vals = [1]
                cols = [ctr]
                bc_ctr=bc_ctr+1
            #print('row',i*j)
            #print(len(vals))
            nnz_save[ctr] = len(vals) 
            Big_K.setValues(ctr,cols,vals)
            ctr=ctr+1
    print(nnz_save)
    print(nnz-nnz_save)
    #Big_K.assemble()
    return Big_K
