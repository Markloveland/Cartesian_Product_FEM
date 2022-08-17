import numpy as np
from fenics import *
import petsc4py
petsc4py.init()
from petsc4py import PETSc
from mpi4py import MPI
from scipy import sparse as sp
from ufl import nabla_div

#Some auxillary functions defined first
def cartesian_product(array1,array2):
    #gives cartesian product of 2 vectors
    dim1,n=array1.shape
    dim2,n=array2.shape
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
def fetch_boundary_dofs(V1,V2,dof_coordinates1,dof_coordinates2):
    #outputs a vector of equation numbers in global system that are on the global boundary
    #input, the two function spaces of the subdomains in the cartesian product
    
    #use this to mark boundary (Dirichlet) 
    def boundary(x, on_boundary):
        return on_boundary

    #This function is simply used to mark the boundary (not the actual boundary condition)
    u_D1 = Expression('1.0', degree=1)

    #establish dummy functions
    dum1=Function(V1)
    dum2=Function(V2)
    #get entire boundary
    bc1 = DirichletBC(V1, u_D1, boundary)
    bc2 = DirichletBC(V2, u_D1, boundary)

    #apply to vectors to mark boundaries
    bc1.apply(dum1.vector())
    bc2.apply(dum2.vector())

    #get index number for each boundary cooordinates in subdomain
    boundary_dofs1 = np.where(dum1.vector()==1.0)[0]
    boundary_dofs2 = np.where(dum2.vector()==1.0)[0]
  
    #coordinates of boundary in each subdomain (just for checking)
    boundary_coord1 = dof_coordinates1[boundary_dofs1]
    boundary_coord2 = dof_coordinates2[boundary_dofs2]

    #now create array of global boundary dof numbers
    global_boundary_dofs=np.empty((len(boundary_dofs1)*len(dof_coordinates2) + len(dof_coordinates1)*len(boundary_dofs2),2))

    ctr=0
    for j in boundary_dofs1:
        global_boundary_dofs[ctr*len(dof_coordinates2):(ctr+1)*len(dof_coordinates2),:] = \
        cartesian_product_dofs(np.array([j]),np.arange(dof_coordinates2.shape[0]))
        ctr=ctr+1

    last_ind = (ctr)*len(dof_coordinates2)


    for j in boundary_dofs2:
        global_boundary_dofs[last_ind:last_ind+len(dof_coordinates1),:] = \
        cartesian_product_dofs(np.arange(dof_coordinates1.shape[0]),np.array([j]))
        last_ind = last_ind+len(dof_coordinates1)    
    
    #sorts and also eliminates duplicates of "corners"
    if len(global_boundary_dofs) !=0:
        global_boundary_dofs=np.unique(global_boundary_dofs,axis=0)

    #have cartesian product of dof at entire boundary (this form should be easy to get coordinates in if needed)
    #now need to convert to global system dof as the kron function does
    global_boundary_dofs=cartesian_form_to_kroneck_form(global_boundary_dofs, len(dof_coordinates2))
    global_boundary_dofs=global_boundary_dofs.astype("int32")
    return global_boundary_dofs

def assemble_global_CSR(Arow,Acol,Brow,Bcol,dat):
    #assembles inputs to load PETSc CSR matrix
    #option 1 is taking domain 2 matrices and putting in 
    #block by block

    #optiom 2 is taking domain 1 matrices and putting in
    #piece by piece

    nnzA = Arow[1:] - Arow[:-1]
    nnzB = Brow[1:] - Brow[:-1]
    nA = len(nnzA)
    nB = len(nnzB)
    #print('Num A')
    #print(len(nnzA))
    #print('Num B')
    #print(len(nnzB))
    Kcol = np.zeros(dat.size)
    Krow = np.zeros(nA*nB+1)
    Kdat = np.zeros(dat.size)
        
    ind = 0
    ctr = 0
    j0 = 0
    for i in range(nA):
        n1 = nnzA[i]
        k0=0
        for k in range(nB):
            n2 = nnzB[k]
            for j in range(n1):
                #print(n2)
                #print(Kdat[ctr:ctr+n2].shape)
                #print(dat[k0:k0+n2,j0+j].shape)
                Kdat[ctr:ctr+n2] = dat[k0:k0+n2,j0+j]
                Kcol[ctr:ctr+n2] = np.array(Acol[j0+j]*nB)+Bcol[k0:k0+n2]
                ctr=ctr+n2
            ind=ind+1
            Krow[ind]=ctr
            k0=k0+n2
        j0=j0+n1

    return Krow,Kcol,Kdat
def create_cartesian_mass_matrix(local_rows,global_rows,local_cols,global_cols):
    #Allocate global mass matrix
    #need to generate global mass matrix to get global matrix layout and sparsity patterns
    #first I need to create an mpi matrix of the appropriate size and start storing values
    M = PETSc.Mat()
    M.create(comm=MPI.COMM_WORLD)
    M.setSizes(([local_rows,global_rows],[local_cols,global_cols]),bsize=1)
    M.setFromOptions()
    M.setType('aij')
    M.setUp()
    #also need global stiffness matrix
    #same exact structure as M
    return M

def build_cartesian_mass_matrix(M1_pet,M2_pet,M1_sizes,M1_global_size,M2_sizes,M):
    #takes in 2 Fenics tensors and creates PETSc matrix entries in
    #CSR format
    ###################################################################
    #Preallocate!
    #to get nnz of cartesian matrix, need nnz of each submatrix
    #use mass matrix to specify sparsity pattern
    M1_I,M1_J,M1_A = M1_pet.mat().getValuesCSR()
    M1_NNZ = M1_I[1:]-M1_I[:-1]
    M2_I,M2_J,M2_A = M2_pet.mat().getValuesCSR()
    M2_NNZ = M2_I[1:]-M2_I[:-1]
    M1 = sp.csr_matrix((M1_A,M1_J,M1_I),shape=(M1_sizes[0],M1_global_size[1]))
    M2 = sp.csr_matrix((M2_A,M2_J,M2_I),shape=M2_sizes)
    temp2 = sp.kron(M1,M2,format="csr")
    M_NNZ = temp2.indptr[1:]-temp2.indptr[:-1]

    #now need to mass matrixes for stiffness and RHS
    ##################################################################
    ##################################################################
    #Loading routine for mass and stiffness matrix
    #set the global mass matrix using CSR
    M.setValuesCSR(temp2.indptr,temp2.indices,temp2.data)
    M.assemble()
    return M_NNZ

def build_stiffness_varying_poisson(V1,V2,kappa,N_dof_2,A):
    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)
    u2 = TrialFunction(V2)
    v2 = TestFunction(V2)
    kappa_func = Function(V1)
    #this stores functions to be intergrated
    #this is terribly inneficient, maybe look into scipy sparse
    #or something like this

    #loop through dof_2 and get a N_dof_1xN_dof_1 sparse matrix
    #each matrix will have same sparsity pattern so get first one then
    #create numpy to store vals

    #need value at a specific dof_coordinate in second domain
    kappa_func.vector()[:] = np.array(kappa[0::N_dof_2])
    #create expressions and assemble linear forms
    K11 = kappa_func*u1.dx(0)*v1.dx(0)*dx
    K12 = kappa_func*u1*v1*dx
    #then save all matrices to list of matrices
    #since these are sparse maybe take PETSc output and pipe
    #to scipy sparse matrices
    #maybe not so easy to program on second loop though

    #K1,K2 are temporary variables to store matrices
    K1=PETScMatrix()
    K2 = PETScMatrix()
    assemble(K11,tensor=K1)
    assemble(K12,tensor=K2)

    #store sparsity pattern (rows,columns, vals)
    A1_I,A1_J,temp = K1.mat().getValuesCSR()
    A2_I,A2_J,temp2 = K2.mat().getValuesCSR()
    len1 = len(temp)
    len2 = len(temp2)
    #create np to store N_dof_2 sets of vals
    vals1 = np.zeros((len1,N_dof_2))
    vals2 = np.zeros((len2,N_dof_2))
    vals1[:,0] = temp
    vals2[:,0] = temp2
    #need to loop over nodes in N-dof-2
    for a in range(1,N_dof_2):
        #need value at a specific dof_coordinate in second domain
        kappa_func.vector()[:] = np.array(kappa[a::N_dof_2])
        #create expressions and assemble linear forms
        K11 = kappa_func*u1.dx(0)*v1.dx(0)*dx
        K12 = kappa_func*u1*v1*dx
        #then save all matrices to list of matrices
        #since these are sparse maybe take PETSc output and pipe
        #to scipy sparse matrices
        #maybe not so easy to program on second loop though

        #need to rebuild each time?
        K1 = PETScMatrix()
        K2 = PETScMatrix()
        assemble(K11,tensor=K1)
        assemble(K12,tensor=K2)


        _,_,temp = K1.mat().getValuesCSR()
        _,_,temp2 = K2.mat().getValuesCSR()

        vals1[:,a] = temp
        vals2[:,a] = temp2


    #now for each entry in sparse N_dof_1 x N_dof_1 matrix need to evaluate
    # int_Omega2 fy ... dy
    #like before, first need to get sparsity patterns


    fy = Function(V2)

    fy.vector()[:] = np.array(vals1[0,:])

    K1 = PETScMatrix(MPI.COMM_SELF)
    K21 = u2*v2*fy*dx
    assemble(K21,tensor=K1)


    K2 = PETScMatrix(MPI.COMM_SELF)
    fy.vector()[:] = np.array(vals2[0,:])
    K22 = u2.dx(0)*v2.dx(0)*fy*dx
    assemble(K22,tensor=K2)


    B1_I,B1_J,temp = K1.mat().getValuesCSR()
    B2_I,B2_J,temp2 = K2.mat().getValuesCSR()
    #print('B1_I')
    #print(B1_I)
    #print(B1_J)
    #print(temp)


    blen1 = len(temp)
    blen2 = len(temp2)

    dat1 = np.zeros((blen1,len1))
    dat2 = np.zeros((blen2,len2))

    dat1[:,0] = temp
    dat2[:,0] = temp2

    #KEY! IDK IF TRUE BUT ASSUMING length of sparse matrixes K1,K2 were same
    #If not, then will need separate loops

    for i in range(1,len1):
        fy.vector()[:] = np.array(vals1[i,:])

        K1 = PETScMatrix(MPI.COMM_SELF)
        K21 = u2*v2*fy*dx
        assemble(K21,tensor=K1)


        K2 = PETScMatrix(MPI.COMM_SELF)
        fy.vector()[:] = np.array(vals2[i,:])
        K22 = u2.dx(0)*v2.dx(0)*fy*dx
        assemble(K22,tensor=K2)

        _,_,temp = K1.mat().getValuesCSR()
        _,_,temp2 = K2.mat().getValuesCSR()

        dat1[:,i] = temp
        dat2[:,i] = temp2


    Krow,Kcol,Kdat = assemble_global_CSR(A1_I,A1_J,B1_I,B1_J,dat1)
    Krow2,Kcol2,Kdat2 = assemble_global_CSR(A2_I,A2_J,B2_I,B2_J,dat2)
    #lastly need to rearrange indeces and rows to give final assignment in A

    #see if sparsity patterns are identical
    #print(np.sum(Kcol-Kcol2))
    #print(np.sum(Krow-Krow2))

    Krow=Krow.astype(np.int32)
    Kcol=Kcol.astype(np.int32)

    #assign values to PETSc matrix
    A.setValuesCSR(Krow,Kcol,Kdat+Kdat2)
    A.assemble()
    return 0


def build_stiffness_varying_action_balance(mesh1,V1,mesh2,V2,c,N_dof_2,dt,A):
    #Builds and assembles stiffness matrix for action balance equations
    # mesh1 is geographic mesh
    # V1 is FunctionSpace for geographic mesh
    # mesh2 is spectral mesh
    # V2 is FunctionSpace for spectral mesh
    # c is group velocity vector at each d.o.f: should be #dof x 4 np vector
    # N_dof_2 is number of degrees of freedom in spectral domain
    # A is PETSc Matrix which is already preallocated, where output will be loaded

    n1 = FacetNormal(mesh1)
    n2 = FacetNormal(mesh2)
    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)
    u2 = TrialFunction(V2)
    v2 = TestFunction(V2)
    cx_func = Function(V1)
    cy_func = Function(V1)
    csig_func = Function(V1)
    cthet_func = Function(V1)

    #loop through dof_2 and get a N_dof_1xN_dof_1 sparse matrix
    #each matrix will have same sparsity pattern so get first one then
    #create numpy to store vals
    A_global_size = A.getSize()
    A_local_size = A.getLocalSize()

    #need value at a specific dof_coordinate in second domain
    cx_func.vector()[:] = np.array(c[0::N_dof_2,0])
    cy_func.vector()[:] = np.array(c[0::N_dof_2,1])
    csig_func.vector()[:] = np.array(c[0::N_dof_2,2])
    cthet_func.vector()[:] = np.array(c[0::N_dof_2,3])
    #create expressions and assemble linear forms
    K11 = cx_func*u1*v1.dx(0)*dx + cy_func*u1*v1.dx(1)*dx
    K12 = csig_func*u1*v1*dx
    K13 = cthet_func*u1*v1*dx
    K14 = dot(as_vector((cx_func,cy_func)),n1)*u1*v1*ds
    #K14 = u1*v1*ds
    #then save all matrices to list of matrices
    #since these are sparse maybe take PETSc output and pipe
    #to scipy sparse matrices
    #maybe not so easy to program on second loop though

    #K1,K2 are temporary variables to store matrices
    # IS THIS BEST WAY? OR SHOULD I REWRITE ONLY 1 MATRIX EACH TIME???
    K1 = PETScMatrix()
    K2 = PETScMatrix()
    K3 = PETScMatrix()
    K4 = PETScMatrix()
    assemble(K11,tensor=K1)
    assemble(K12,tensor=K2)
    assemble(K13,tensor=K3)
    assemble(K14,tensor=K4)
    
    #store sparsity pattern (rows,columns, vals)
    A1_I,A1_J,temp = K1.mat().getValuesCSR()
    A2_I,A2_J,temp2 = K2.mat().getValuesCSR()
    A3_I,A3_J,temp3 = K3.mat().getValuesCSR()
    A4_I,A4_J,temp4 = K4.mat().getValuesCSR()
    len1 = len(temp)
    len2 = len(temp2)
    len3 = len(temp3)
    len4 = len(temp4)
    
    #create np to store N_dof_2 sets of vals
    vals1 = np.zeros((len1,N_dof_2))
    vals2 = np.zeros((len2,N_dof_2))
    vals3 = np.zeros((len3,N_dof_2))
    vals4 = np.zeros((len4,N_dof_2))
    vals1[:,0] = temp
    vals2[:,0] = temp2
    vals3[:,0] = temp3
    vals4[:,0] = temp4
    #need to loop over nodes in N-dof-2
    for a in range(1,N_dof_2):
        cx_func.vector()[:] = np.array(c[a::N_dof_2,0])
        cy_func.vector()[:] = np.array(c[a::N_dof_2,1])
        csig_func.vector()[:] = np.array(c[a::N_dof_2,2])
        cthet_func.vector()[:] = np.array(c[a::N_dof_2,3])
        #create expressions and assemble linear forms
        K11 = cx_func*u1*v1.dx(0)*dx + cy_func*u1*v1.dx(1)*dx
        K12 = csig_func*u1*v1*dx
        K13 = cthet_func*u1*v1*dx
        K14 = dot(as_vector((cx_func,cy_func)),n1)*u1*v1*ds
        # IS THIS BEST WAY? OR SHOULD I REWRITE ONLY 1 MATRIX EACH TIME???
        K1 = PETScMatrix()
        K2 = PETScMatrix()
        K3 = PETScMatrix()
        K4 = PETScMatrix()
        assemble(K11,tensor=K1)
        assemble(K12,tensor=K2)
        assemble(K13,tensor=K3)
        assemble(K14,tensor=K4)

        _,_,temp = K1.mat().getValuesCSR()
        _,_,temp2 = K2.mat().getValuesCSR()
        _,_,temp3 = K3.mat().getValuesCSR()
        _,_,temp4 = K4.mat().getValuesCSR()

        vals1[:,a] = temp
        vals2[:,a] = temp2
        vals3[:,a] = temp3
        vals4[:,a] = temp4

    #now for each entry in sparse N_dof_1 x N_dof_1 matrix need to evaluate
    # int_Omega2 fy ... dy
    #like before, first need to get sparsity patterns


    fy1 = Function(V2)
    fy2 = Function(V2)
    fy3 = Function(V2)

    fy1.vector()[:] = np.array(vals1[0,:])
    fy2.vector()[:] = np.array(vals2[0,:])
    fy3.vector()[:]  = np.array(vals3[0,:])
    

        
    K21 = -u2*v2*fy1*dx
    K21 += -inner(u2*as_vector((fy2,fy3)),grad(v2))*dx
    K21 += u2*v2*dot(as_vector((fy2,fy3)),n2)*ds
    
    
    K1 = PETScMatrix(MPI.COMM_SELF)
    assemble(K21,tensor=K1)


    B1_I,B1_J,temp = K1.mat().getValuesCSR()

    blen1 = len(temp)
    dat1 = np.zeros((blen1,len1)) 
    dat1[:,0] = temp


    #KEY! IDK IF TRUE BUT ASSUMING length of sparse matrixes K1,K2,K3 were same
    #If not, then will need separate loops

    for i in range(1,len1):


        fy1.vector()[:] = np.array(vals1[i,:])
        fy2.vector()[:] = np.array(vals2[i,:])
        fy3.vector()[:]  = np.array(vals3[i,:])
    

        
        K21 = -u2*v2*fy1*dx
        K21 += -inner(u2*as_vector((fy2,fy3)),grad(v2))*dx
        K21 += u2*v2*dot(as_vector((fy2,fy3)),n2)*ds
    
    
        K1 = PETScMatrix(MPI.COMM_SELF)
        assemble(K21,tensor=K1)


        _,_,temp = K1.mat().getValuesCSR()

        dat1[:,i] = temp


    Krow,Kcol,Kdat = assemble_global_CSR(A1_I,A1_J,B1_I,B1_J,dat1)

    Krow=Krow.astype(np.int32)
    Kcol=Kcol.astype(np.int32)
    #challenge is we likely have 3 different sparsity patterns so how should we
    #add them all using scipy???
    K1 = sp.csr_matrix((Kdat, Kcol, Krow), shape=(A_local_size[0],A_global_size[1]))

    #add the sparse matrices
    K = dt*K1
    
    #only works if there is boundary of domain 1 on this process
    if len(vals4[:,0]) != 0:
        K4 = PETScMatrix(MPI.COMM_SELF)
        fy1.vector()[:] = np.array(vals4[0,:])
        K24 = u2*v2*fy1*dx
        assemble(K24,tensor=K4)
        B4_I,B4_J,temp4 = K4.mat().getValuesCSR()
        blen4 = len(temp4)
        dat4 = np.zeros((blen4,len4))
        dat4[:,0] = temp4

        #K4 is the boundary integral dOmega1 x Omega2
        for i in range(1,len4):
            K4 = PETScMatrix(MPI.COMM_SELF)
            fy1.vector()[:] = np.array(vals4[i,:])
            K24 = u2*v2*fy1*dx
            assemble(K24,tensor=K4)

            _,_,temp4 = K4.mat().getValuesCSR()
            dat4[:,i] = temp4
    
    
        Krow4,Kcol4,Kdat4 = assemble_global_CSR(A4_I,A4_J,B4_I,B4_J,dat4)
   
        Krow4=Krow4.astype(np.int32)
        Kcol4=Kcol4.astype(np.int32)

        K3 = sp.csr_matrix((Kdat4, Kcol4, Krow4), shape=(A_local_size[0],A_global_size[1]))

        K = K + dt*K3
    
    #assign values to PETSc matrix
    A.setValuesCSR(K.indptr,K.indices,K.data)
    A.assemble()
    return 0
    
def build_stiffness_varying_action_balance_2D(mesh1,V1,mesh2,V2,c,N_dof_2,dt,A):
    #Builds and assembles stiffness matrix for action balance equations
    # mesh1 is geographic mesh
    # V1 is FunctionSpace for geographic mesh
    # mesh2 is spectral mesh
    # V2 is FunctionSpace for spectral mesh
    # c is group velocity vector at each d.o.f: should be #dof x 4 np vector
    # N_dof_2 is number of degrees of freedom in spectral domain
    # A is PETSc Matrix which is already preallocated, where output will be loaded

    n1 = FacetNormal(mesh1)
    n2 = FacetNormal(mesh2)
    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)
    u2 = TrialFunction(V2)
    v2 = TestFunction(V2)
    cx_func = Function(V1)
    cy_func = Function(V1)

    #loop through dof_2 and get a N_dof_1xN_dof_1 sparse matrix
    #each matrix will have same sparsity pattern so get first one then
    #create numpy to store vals
    A_global_size = A.getSize()
    A_local_size = A.getLocalSize()

    #need value at a specific dof_coordinate in second domain
    cx_func.vector()[:] = np.array(c[0::N_dof_2,0])
    cy_func.vector()[:] = np.array(c[0::N_dof_2,1])
    #create expressions and assemble linear forms
    K11 = cx_func*u1*v1.dx(0)*dx
    K12 = cy_func*u1*v1*dx
    K14 = dot(as_vector((cx_func,)),n1)*u1*v1*ds
    #K14 = u1*v1*ds
    #then save all matrices to list of matrices
    #since these are sparse maybe take PETSc output and pipe
    #to scipy sparse matrices
    #maybe not so easy to program on second loop though

    #K1,K2 are temporary variables to store matrices
    # IS THIS BEST WAY? OR SHOULD I REWRITE ONLY 1 MATRIX EACH TIME???
    K1 = PETScMatrix()
    K2 = PETScMatrix()
    K4 = PETScMatrix()
    assemble(K11,tensor=K1)
    assemble(K12,tensor=K2)
    assemble(K14,tensor=K4)
    
    #store sparsity pattern (rows,columns, vals)
    A1_I,A1_J,temp = K1.mat().getValuesCSR()
    A2_I,A2_J,temp2 = K2.mat().getValuesCSR()
    A4_I,A4_J,temp4 = K4.mat().getValuesCSR()
    len1 = len(temp)
    len2 = len(temp2)
    len4 = len(temp4)
    
    #create np to store N_dof_2 sets of vals
    vals1 = np.zeros((len1,N_dof_2))
    vals2 = np.zeros((len2,N_dof_2))
    vals4 = np.zeros((len4,N_dof_2))
    vals1[:,0] = temp
    vals2[:,0] = temp2
    vals4[:,0] = temp4

    #need to loop over nodes in N-dof-2
    for a in range(1,N_dof_2):
        cx_func.vector()[:] = np.array(c[a::N_dof_2,0])
        cy_func.vector()[:] = np.array(c[a::N_dof_2,1])
        #create expressions and assemble linear forms
        K11 = cx_func*u1*v1.dx(0)*dx
        K12 = cy_func*u1*v1*dx
        K14 = dot(as_vector((cx_func,))*u1,n1)*v1*ds
        # IS THIS BEST WAY? OR SHOULD I REWRITE ONLY 1 MATRIX EACH TIME???
        #K1 = PETScMatrix()
        #K2 = PETScMatrix()
        #K4 = PETScMatrix()
        assemble(K11,tensor=K1)
        assemble(K12,tensor=K2)
        assemble(K14,tensor=K4)

        _,_,temp = K1.mat().getValuesCSR()
        _,_,temp2 = K2.mat().getValuesCSR()
        _,_,temp4 = K4.mat().getValuesCSR()

        vals1[:,a] = temp
        vals2[:,a] = temp2
        vals4[:,a] = temp4
    #now for each entry in sparse N_dof_1 x N_dof_1 matrix need to evaluate
    # int_Omega2 fy ... dy
    #like before, first need to get sparsity patterns


    fy = Function(V2)
    fy2 = Function(V2)
    fy.vector()[:] = np.array(vals1[0,:])
    K1 = PETScMatrix(MPI.COMM_SELF)
    K21 = u2*v2*fy*dx
    assemble(K21,tensor=K1)


    K2 = PETScMatrix(MPI.COMM_SELF)
    fy.vector()[:] = np.array(vals2[0,:])
    K22 = inner(u2*fy,v2.dx(0))*dx
    assemble(K22,tensor=K2)

        
    K3 = PETScMatrix(MPI.COMM_SELF)
    K23 = u2*v2*dot(as_vector((fy,)),n2)*ds
    assemble(K23,tensor=K3)

    

    B1_I,B1_J,temp = K1.mat().getValuesCSR()
    B2_I,B2_J,temp2 = K2.mat().getValuesCSR()
    B3_I,B3_J,temp3 = K3.mat().getValuesCSR()

    blen1 = len(temp)
    blen2 = len(temp2)
    blen3 = len(temp3)
    

    dat1 = np.zeros((blen1,len1))
    dat2 = np.zeros((blen2,len2))
    dat3 = np.zeros((blen3,len2))
    
    dat1[:,0] = temp
    dat2[:,0] = temp2
    dat3[:,0] = temp3

    #KEY! IDK IF TRUE BUT ASSUMING length of sparse matrixes K1,K2,K3 were same
    #If not, then will need separate loops

    for i in range(1,len1):

        fy.vector()[:] = np.array(vals1[i,:])
        #K1 = PETScMatrix(MPI.COMM_SELF)
        K21 = u2*v2*fy*dx
        assemble(K21,tensor=K1)


        #K2 = PETScMatrix(MPI.COMM_SELF)
        fy.vector()[:] = np.array(vals2[i,:])
        K22 = fy*u2*v2.dx(0)*dx
        assemble(K22,tensor=K2)


        #K3 = PETScMatrix(MPI.COMM_SELF)
        K23 = u2*v2*dot(as_vector((fy,)),n2)*ds
        assemble(K23,tensor=K3)


        _,_,temp = K1.mat().getValuesCSR()
        _,_,temp2 = K2.mat().getValuesCSR()
        _,_,temp3 = K3.mat().getValuesCSR()
        dat1[:,i] = temp
        dat2[:,i] = temp2
        dat3[:,i] = temp3
    
    
    Krow,Kcol,Kdat = assemble_global_CSR(A1_I,A1_J,B1_I,B1_J,dat1)
    Krow2,Kcol2,Kdat2 = assemble_global_CSR(A2_I,A2_J,B2_I,B2_J,dat2)
    Krow3,Kcol3,Kdat3 = assemble_global_CSR(A2_I,A2_J,B3_I,B3_J,dat3)

    Krow=Krow.astype(np.int32)
    Kcol=Kcol.astype(np.int32)
    Krow2=Krow2.astype(np.int32)
    Kcol2=Kcol2.astype(np.int32)
    Krow3=Krow3.astype(np.int32)
    Kcol3=Kcol3.astype(np.int32)
    #challenge is we likely have 3 different sparsity patterns so how should we
    #add them all using scipy???
    K1 = sp.csr_matrix((Kdat+Kdat2, Kcol, Krow), shape=(A_local_size[0],A_global_size[1]))
    K2 = sp.csr_matrix((Kdat3, Kcol3, Krow3), shape=(A_local_size[0],A_global_size[1]))

    #add the sparse matrices
    K = dt*(-K1+K2)
    
    #only works if there is boundary of domain 1 on this process
    if len(vals4[:,0]) != 0:
        K4 = PETScMatrix(MPI.COMM_SELF)
        fy.vector()[:] = np.array(vals4[0,:])
        K24 = u2*v2*fy*dx
        assemble(K24,tensor=K4)
        B4_I,B4_J,temp4 = K4.mat().getValuesCSR()
        blen4 = len(temp4)
        dat4 = np.zeros((blen4,len4))
        dat4[:,0] = temp4

        #K4 is the boundary integral dOmega1 x Omega2
        for i in range(1,len4):
            #K4 = PETScMatrix(MPI.COMM_SELF)
            fy.vector()[:] = np.array(vals4[i,:])
            K24 = u2*v2*fy*dx
            assemble(K24,tensor=K4)

            _,_,temp4 = K4.mat().getValuesCSR()
            dat4[:,i] = temp4
    
    
        Krow4,Kcol4,Kdat4 = assemble_global_CSR(A4_I,A4_J,B4_I,B4_J,dat4)
   
        Krow4=Krow4.astype(np.int32)
        Kcol4=Kcol4.astype(np.int32)

        K3 = sp.csr_matrix((Kdat4, Kcol4, Krow4), shape=(A_local_size[0],A_global_size[1]))

        K = K + dt*K3

    #assign values to PETSc matrix
    A.setValuesCSR(K.indptr,K.indices,K.data)
    A.assemble()
    return 0

def compute_wave_speeds(x,y,sigma,theta,depth,u,v,g=9.81):
    #need a function to calculate wave speed (phase and group) and wavenumber
    #takes in degrees of freedom and computes wave speeds pointwise
    N_dof = len(sigma)
    c_out = np.zeros((N_dof,4))
    temp = np.zeros(N_dof)
    k = np.zeros(N_dof)
    #employ 3 different approximations depending on water depth
    WGD=np.sqrt(depth/g)*g
    SND=sigma*np.sqrt(depth/g)

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
    
    #store the c_g (group velocity) as temporary variables
    temp[shallow_range]=cg_shallow(WGD[shallow_range])
    temp[mid_range],k[mid_range]=cg_mid(SND[mid_range],g,depth[mid_range],sigma[mid_range])
    temp[deep_range]=cg_deep(g,sigma[deep_range])
    #save these values in c_out and multiply by appropriate angles
    c_out[:,0] = temp*np.cos(theta)
    c_out[:,1] = temp*np.sin(theta)


    #now calculate wavenumber k and store temporarily
    k[shallow_range]=SND[shallow_range]/depth[shallow_range]
    k[deep_range]=sigma[deep_range]**2/g

    #now calculate c_sigma and c_theta, these are a bit more tricky

    #for now assuming H is constant in time but can fix this later
    #need to use FEniCS to calculate this!
    dHdt=0.0
    dHdy = 0.0
    dudy = 0.0
    dvdx = 0.0  #might not have to be 0, well see
    dvdy = 0.0

    #calc gradient of H w.r.t. x
    #this is just forward euler but only works for fixed geometry
    #instead we'll hard code for this case
    dHdx=-1.0/200.0
    dudx=0.0

    #now calculate velocity vectors
    #c_sigma
    c_out[:,2] = k*sigma/(np.sinh(2*k*depth)) *(dHdt + u*dHdx + v*dHdy) - temp*k*(dudx)
    #c theta
    c_out[:,3] = sigma/(np.sinh(2*k*depth))*(dHdx*np.sin(theta)- dHdy*np.cos(theta)) + \
        dudx*np.cos(theta)*np.sin(theta) - dudy*(np.cos(theta)**2) + dvdx*(np.sin(theta)**2) \
        -dvdy*np.cos(theta)*np.sin(theta)
    return c_out


def calculate_HS(u_cart,V2,N_dof_1,N_dof_2):
    HS_vec = np.zeros(N_dof_1)
    dum = Function(V2)
    for i in range(N_dof_1):
        indx = i*N_dof_2
        dum.vector()[:] = np.array(u_cart.getArray()[indx:indx+N_dof_2])
        HS_vec[i] = 4*np.sqrt(abs(assemble(dum*dx)))
    return HS_vec


def peval(f, x,comm):
    '''Parallel synced eval'''
    try:
        yloc = f(x)
    except RuntimeError:
        yloc = np.inf*np.ones(1)

    yglob = np.zeros_like(yloc)
    comm.Reduce(yloc, yglob, op=MPI.MIN)
    return yglob



def build_stiffness_varying_action_balance_SUPG(mesh1,V1,mesh2,V2,c,N_dof_1,N_dof_2,dt,A):
    #Builds and assembles stiffness matrix for action balance equations
    # mesh1 is geographic mesh
    # V1 is FunctionSpace for geographic mesh
    # mesh2 is spectral mesh
    # V2 is FunctionSpace for spectral mesh
    # c is group velocity vector at each d.o.f: should be #dof x 4 np vector
    # N_dof_2 is number of degrees of freedom in spectral domain
    # A is PETSc Matrix which is already preallocated, where output will be loaded

    n1 = FacetNormal(mesh1)
    n2 = FacetNormal(mesh2)
    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)
    u2 = TrialFunction(V2)
    v2 = TestFunction(V2)
    cx_func = Function(V1)
    cy_func = Function(V1)
    csig_func = Function(V1)
    cthet_func = Function(V1)

    #loop through dof_2 and get a N_dof_1xN_dof_1 sparse matrix
    #each matrix will have same sparsity pattern so get first one then
    #create numpy to store vals
    A_global_size = A.getSize()
    A_local_size = A.getLocalSize()

    #need value at a specific dof_coordinate in second domain
    cx_func.vector()[:] = np.array(c[0::N_dof_2,0])
    cy_func.vector()[:] = np.array(c[0::N_dof_2,1])
    csig_func.vector()[:] = np.array(c[0::N_dof_2,2])
    cthet_func.vector()[:] = np.array(c[0::N_dof_2,3])
    #create expressions and assemble linear forms
    K11 = cx_func*u1*v1.dx(0)*dx + cy_func*u1*v1.dx(1)*dx
    K12 = csig_func*u1*v1*dx
    K13 = cthet_func*u1*v1*dx
    K14 = dot(as_vector((cx_func,cy_func)),n1)*u1*v1*ds
    #

    #SUPG stabilization
    R =  Circumradius(mesh1)*mesh2.hmax()
    tau = R/(cx_func**2 + cy_func**2 + csig_func**2 + cthet_func**2)**0.5
    c1 = as_vector((cx_func,cy_func))
    c2 = as_vector((csig_func,cthet_func))
    SUPG1 = tau*dot(c1,grad(u1))*dot(c1,grad(v1))*dx
    SUPG2 = tau*csig_func*u1*dot(c1,grad(v1))*dx
    SUPG3 = tau*cthet_func*u1*dot(c1,grad(v1))*dx
    SUPG4 = tau*u1*nabla_div(c1)*dot(c1,grad(v1))*dx
    SUPG5 = tau*csig_func*v1*dot(c1,grad(u1))*dx
    SUPG6 = tau*cthet_func*v1*dot(c1,grad(u1))*dx
    SUPG7 = tau*csig_func*v1*u1*nabla_div(c1)*dx
    SUPG8 = tau*cthet_func*v1*u1*nabla_div(c1)*dx

    #then save all matrices to list of matrices
    #since these are sparse maybe take PETSc output and pipe
    #to scipy sparse matrices
    #maybe not so easy to program on second loop though

    #K1,K2 are temporary variables to store matrices
    # IS THIS BEST WAY? OR SHOULD I REWRITE ONLY 1 MATRIX EACH TIME???
    K1 = PETScMatrix()
    K2 = PETScMatrix()
    K3 = PETScMatrix()
    K4 = PETScMatrix()
    assemble(K11,tensor=K1)
    assemble(K12,tensor=K2)
    assemble(K13,tensor=K3)
    assemble(K14,tensor=K4)
    
    #store sparsity pattern (rows,columns, vals)
    A1_I,A1_J,temp = K1.mat().getValuesCSR()
    A2_I,A2_J,temp2 = K2.mat().getValuesCSR()
    A3_I,A3_J,temp3 = K3.mat().getValuesCSR()
    A4_I,A4_J,temp4 = K4.mat().getValuesCSR()
    len1 = len(temp)
    len2 = len(temp2)
    len3 = len(temp3)
    len4 = len(temp4)
    
    #create np to store N_dof_2 sets of vals
    vals1 = np.zeros((len1,N_dof_2))
    vals2 = np.zeros((len2,N_dof_2))
    vals3 = np.zeros((len3,N_dof_2))
    vals4 = np.zeros((len4,N_dof_2))
    vals1[:,0] = temp
    vals2[:,0] = temp2
    vals3[:,0] = temp3
    vals4[:,0] = temp4



    ##SUPG Terms#########
    SUPG_1 = PETScMatrix()
    SUPG_2 = PETScMatrix()
    SUPG_3 = PETScMatrix()
    SUPG_4 = PETScMatrix()
    SUPG_5 = PETScMatrix()
    SUPG_6 = PETScMatrix()
    SUPG_7 = PETScMatrix()
    SUPG_8 = PETScMatrix()
    assemble(SUPG1,tensor=SUPG_1)
    assemble(SUPG2,tensor=SUPG_2)
    assemble(SUPG3,tensor=SUPG_3)
    assemble(SUPG4,tensor=SUPG_4)
    assemble(SUPG5,tensor=SUPG_5)
    assemble(SUPG6,tensor=SUPG_6)
    assemble(SUPG7,tensor=SUPG_7)
    assemble(SUPG8,tensor=SUPG_8)

    SUPG1_I,SUPG1_J,SUPG1_temp = SUPG_1.mat().getValuesCSR()
    SUPG2_I,SUPG2_J,SUPG2_temp = SUPG_2.mat().getValuesCSR()
    SUPG3_I,SUPG3_J,SUPG3_temp = SUPG_3.mat().getValuesCSR()
    SUPG4_I,SUPG4_J,SUPG4_temp = SUPG_4.mat().getValuesCSR()
    SUPG5_I,SUPG5_J,SUPG5_temp = SUPG_5.mat().getValuesCSR()
    SUPG6_I,SUPG6_J,SUPG6_temp = SUPG_6.mat().getValuesCSR()
    SUPG7_I,SUPG7_J,SUPG7_temp = SUPG_7.mat().getValuesCSR()
    SUPG8_I,SUPG8_J,SUPG8_temp = SUPG_8.mat().getValuesCSR()

    SUPG1_len = len(SUPG1_temp)
    SUPG2_len = len(SUPG2_temp)
    SUPG3_len = len(SUPG3_temp)
    SUPG4_len = len(SUPG4_temp)
    SUPG5_len = len(SUPG5_temp)
    SUPG6_len = len(SUPG6_temp)
    SUPG7_len = len(SUPG7_temp)
    SUPG8_len = len(SUPG8_temp)

    SUPG1_vals = np.zeros((SUPG1_len,N_dof_2))
    SUPG2_vals = np.zeros((SUPG2_len,N_dof_2))
    SUPG3_vals = np.zeros((SUPG3_len,N_dof_2))
    SUPG4_vals = np.zeros((SUPG4_len,N_dof_2))
    SUPG5_vals = np.zeros((SUPG5_len,N_dof_2))
    SUPG6_vals = np.zeros((SUPG6_len,N_dof_2))
    SUPG7_vals = np.zeros((SUPG7_len,N_dof_2))
    SUPG8_vals = np.zeros((SUPG8_len,N_dof_2))

    SUPG1_vals[:,0] = SUPG1_temp
    SUPG2_vals[:,0] = SUPG2_temp
    SUPG3_vals[:,0] = SUPG3_temp
    SUPG4_vals[:,0] = SUPG4_temp
    SUPG5_vals[:,0] = SUPG5_temp
    SUPG6_vals[:,0] = SUPG6_temp
    SUPG7_vals[:,0] = SUPG7_temp
    SUPG8_vals[:,0] = SUPG8_temp
    ###########################

    #need to loop over nodes in N-dof-2
    for a in range(1,N_dof_2):
        cx_func.vector()[:] = np.array(c[a::N_dof_2,0])
        cy_func.vector()[:] = np.array(c[a::N_dof_2,1])
        csig_func.vector()[:] = np.array(c[a::N_dof_2,2])
        cthet_func.vector()[:] = np.array(c[a::N_dof_2,3])
        c1 = as_vector((cx_func,cy_func))
        c2 = as_vector((csig_func,cthet_func))
        #create expressions and assemble linear forms
        K11 = cx_func*u1*v1.dx(0)*dx + cy_func*u1*v1.dx(1)*dx
        K12 = csig_func*u1*v1*dx
        K13 = cthet_func*u1*v1*dx
        K14 = dot(as_vector((cx_func,cy_func)),n1)*u1*v1*ds
        # IS THIS BEST WAY? OR SHOULD I REWRITE ONLY 1 MATRIX EACH TIME???
        K1 = PETScMatrix()
        K2 = PETScMatrix()
        K3 = PETScMatrix()
        K4 = PETScMatrix()
        assemble(K11,tensor=K1)
        assemble(K12,tensor=K2)
        assemble(K13,tensor=K3)
        assemble(K14,tensor=K4)

        _,_,temp = K1.mat().getValuesCSR()
        _,_,temp2 = K2.mat().getValuesCSR()
        _,_,temp3 = K3.mat().getValuesCSR()
        _,_,temp4 = K4.mat().getValuesCSR()

        vals1[:,a] = temp
        vals2[:,a] = temp2
        vals3[:,a] = temp3
        vals4[:,a] = temp4


        ###############SUPG###################
        
        tau = R/(cx_func**2 + cy_func**2 + csig_func**2 + cthet_func**2)**0.5
        SUPG1 = tau*dot(c1,grad(u1))*dot(c1,grad(v1))*dx
        SUPG2 = tau*csig_func*u1*dot(c1,grad(v1))*dx
        SUPG3 = tau*cthet_func*u1*dot(c1,grad(v1))*dx
        SUPG4 = tau*u1*nabla_div(c1)*dot(c1,grad(v1))*dx
        SUPG5 = tau*csig_func*v1*dot(c1,grad(u1))*dx
        SUPG6 = tau*cthet_func*v1*dot(c1,grad(u1))*dx
        SUPG7 = tau*csig_func*v1*u1*nabla_div(c1)*dx
        SUPG8 = tau*cthet_func*v1*u1*nabla_div(c1)*dx
        SUPG_1 = PETScMatrix()
        SUPG_2 = PETScMatrix()
        SUPG_3 = PETScMatrix()
        SUPG_4 = PETScMatrix()
        SUPG_5 = PETScMatrix()
        SUPG_6 = PETScMatrix()
        SUPG_7 = PETScMatrix()
        SUPG_8 = PETScMatrix()
        assemble(SUPG1,tensor=SUPG_1)
        assemble(SUPG2,tensor=SUPG_2)
        assemble(SUPG3,tensor=SUPG_3)
        assemble(SUPG4,tensor=SUPG_4)
        assemble(SUPG5,tensor=SUPG_5)
        assemble(SUPG6,tensor=SUPG_6)
        assemble(SUPG7,tensor=SUPG_7)
        assemble(SUPG8,tensor=SUPG_8)

        _,_,SUPG1_temp = SUPG_1.mat().getValuesCSR()
        _,_,SUPG2_temp = SUPG_2.mat().getValuesCSR()
        _,_,SUPG3_temp = SUPG_3.mat().getValuesCSR()
        _,_,SUPG4_temp = SUPG_4.mat().getValuesCSR()
        _,_,SUPG5_temp = SUPG_5.mat().getValuesCSR()
        _,_,SUPG6_temp = SUPG_6.mat().getValuesCSR()
        _,_,SUPG7_temp = SUPG_7.mat().getValuesCSR()
        _,_,SUPG8_temp = SUPG_8.mat().getValuesCSR()
    
    
        SUPG1_vals[:,a] = SUPG1_temp
        SUPG2_vals[:,a] = SUPG2_temp
        SUPG3_vals[:,a] = SUPG3_temp
        SUPG4_vals[:,a] = SUPG4_temp
        SUPG5_vals[:,a] = SUPG5_temp
        SUPG6_vals[:,a] = SUPG6_temp
        SUPG7_vals[:,a] = SUPG7_temp
        SUPG8_vals[:,a] = SUPG8_temp
    
    
    #now for each entry in sparse N_dof_1 x N_dof_1 matrix need to evaluate
    # int_Omega2 fy ... dy
    #like before, first need to get sparsity patterns


    fy = Function(V2)
    fy2 = Function(V2)
    fy.vector()[:] = np.array(vals1[0,:])
    K1 = PETScMatrix(MPI.COMM_SELF)
    K21 = u2*v2*fy*dx
    assemble(K21,tensor=K1)


    K2 = PETScMatrix(MPI.COMM_SELF)
    fy.vector()[:] = np.array(vals2[0,:])
    fy2.vector()[:]  = np.array(vals3[0,:])
    K22 = inner(u2*as_vector((fy,fy2)),grad(v2))*dx
    assemble(K22,tensor=K2)

        
    K3 = PETScMatrix(MPI.COMM_SELF)
    K23 = u2*v2*dot(as_vector((fy,fy2)),n2)*ds
    assemble(K23,tensor=K3)

    

    B1_I,B1_J,temp = K1.mat().getValuesCSR()
    B2_I,B2_J,temp2 = K2.mat().getValuesCSR()
    B3_I,B3_J,temp3 = K3.mat().getValuesCSR()

    blen1 = len(temp)
    blen2 = len(temp2)
    blen3 = len(temp3)

    dat1 = np.zeros((blen1,len1))
    dat2 = np.zeros((blen2,len2))
    dat3 = np.zeros((blen3,len3))
    
    dat1[:,0] = temp
    dat2[:,0] = temp2
    dat3[:,0] = temp3

    

    ###########SUPG################
    ######
    SUPG_B1 = PETScMatrix(MPI.COMM_SELF)
    fy.vector()[:] = np.array(SUPG1_vals[0,:])
    fy2.vector()[:] = np.array(SUPG4_vals[0,:])
    SUPGB1 = u2*v2*(fy+fy2)*dx
    assemble(SUPGB1,tensor=SUPG_B1)


    SUPG_B2 = PETScMatrix(MPI.COMM_SELF)
    fy.vector()[:] = np.array(SUPG2_vals[0,:])
    fy2.vector()[:] = np.array(SUPG3_vals[0,:])
    SUPGB2 = v2*dot(grad(u2),as_vector((fy,fy2)))*dx
    assemble(SUPGB2,tensor=SUPG_B2)

    SUPG_B3 = PETScMatrix(MPI.COMM_SELF)
    fy.vector()[:] = np.array(SUPG5_vals[0,:])
    fy2.vector()[:] = np.array(SUPG6_vals[0,:])
    SUPGB3 = u2*dot(grad(v2),as_vector((fy,fy2)))*dx
    assemble(SUPGB3,tensor=SUPG_B3)

    SUPG_B4 = PETScMatrix(MPI.COMM_SELF)
    fy.vector()[:] = np.array(SUPG7_vals[0,:])
    fy2.vector()[:] = np.array(SUPG8_vals[0,:])
    SUPGB4 = u2*dot(grad(v2),as_vector((fy,fy2)))*dx
    assemble(SUPGB4,tensor=SUPG_B4)
    
    SUPGB1_I, SUPGB1_J, SUPGB1_temp = SUPG_B1.mat().getValuesCSR()
    SUPGB2_I, SUPGB2_J, SUPGB2_temp = SUPG_B2.mat().getValuesCSR()
    SUPGB3_I, SUPGB3_J, SUPGB3_temp = SUPG_B3.mat().getValuesCSR()
    SUPGB4_I, SUPGB4_J, SUPGB4_temp = SUPG_B4.mat().getValuesCSR()


    SUPGB1_len = len(SUPGB1_temp)
    SUPGB2_len = len(SUPGB2_temp)
    SUPGB3_len = len(SUPGB3_temp)
    SUPGB4_len = len(SUPGB4_temp)


    SUPG_dat1 = np.zeros((SUPGB1_len,SUPG1_len))
    SUPG_dat2 = np.zeros((SUPGB2_len,SUPG2_len))
    SUPG_dat3 = np.zeros((SUPGB3_len,SUPG3_len))
    SUPG_dat4 = np.zeros((SUPGB4_len,SUPG4_len))

    SUPG_dat1[:,0] = SUPGB1_temp
    SUPG_dat2[:,0] = SUPGB2_temp
    SUPG_dat3[:,0] = SUPGB3_temp
    SUPG_dat4[:,0] = SUPGB4_temp
    #KEY! IDK IF TRUE BUT ASSUMING length of sparse matrixes K1,K2,K3 were same
    #If not, then will need separate loops

    for i in range(1,len1):

        fy.vector()[:] = np.array(vals1[i,:])
        K1 = PETScMatrix(MPI.COMM_SELF)
        K21 = u2*v2*fy*dx
        assemble(K21,tensor=K1)


        K2 = PETScMatrix(MPI.COMM_SELF)
        fy.vector()[:] = np.array(vals2[i,:])
        fy2.vector()[:]  = np.array(vals3[i,:])
        K22 = inner(u2*as_vector((fy,fy2)),grad(v2))*dx
        assemble(K22,tensor=K2)


        K3 = PETScMatrix(MPI.COMM_SELF)
        K23 = u2*v2*dot(as_vector((fy,fy2)),n2)*ds
        assemble(K23,tensor=K3)


        _,_,temp = K1.mat().getValuesCSR()
        _,_,temp2 = K2.mat().getValuesCSR()
        _,_,temp3 = K3.mat().getValuesCSR()
        dat1[:,i] = temp
        dat2[:,i] = temp2
        dat3[:,i] = temp3


        ##########SUPG########

        SUPG_B1 = PETScMatrix(MPI.COMM_SELF)
        fy.vector()[:] = np.array(SUPG1_vals[i,:])
        fy2.vector()[:] = np.array(SUPG4_vals[i,:])
        SUPGB1 = u2*v2*(fy+fy2)*dx
        assemble(SUPGB1,tensor=SUPG_B1)


        SUPG_B2 = PETScMatrix(MPI.COMM_SELF)
        fy.vector()[:] = np.array(SUPG2_vals[i,:])
        fy2.vector()[:] = np.array(SUPG3_vals[i,:])
        SUPGB2 = v2*dot(grad(u2),as_vector((fy,fy2)))*dx
        assemble(SUPGB2,tensor=SUPG_B2)

        SUPG_B3 = PETScMatrix(MPI.COMM_SELF)
        fy.vector()[:] = np.array(SUPG5_vals[i,:])
        fy2.vector()[:] = np.array(SUPG6_vals[i,:])
        SUPGB3 = u2*dot(grad(v2),as_vector((fy,fy2)))*dx
        assemble(SUPGB3,tensor=SUPG_B3)

        SUPG_B4 = PETScMatrix(MPI.COMM_SELF)
        fy.vector()[:] = np.array(SUPG7_vals[i,:])
        fy2.vector()[:] = np.array(SUPG8_vals[i,:])
        SUPGB4 = u2*dot(grad(v2),as_vector((fy,fy2)))*dx
        assemble(SUPGB4,tensor=SUPG_B4)

        _,_,SUPGB1_temp = SUPG_B1.mat().getValuesCSR()
        _,_,SUPGB2_temp = SUPG_B2.mat().getValuesCSR()
        _,_,SUPGB3_temp = SUPG_B3.mat().getValuesCSR()
        _,_,SUPGB4_temp = SUPG_B4.mat().getValuesCSR()


        SUPG_dat1[:,i] = SUPGB1_temp
        SUPG_dat2[:,i] = SUPGB2_temp
        SUPG_dat3[:,i] = SUPGB3_temp
        SUPG_dat4[:,i] = SUPGB4_temp

    Krow,Kcol,Kdat = assemble_global_CSR(A1_I,A1_J,B1_I,B1_J,dat1)
    Krow2,Kcol2,Kdat2 = assemble_global_CSR(A2_I,A2_J,B2_I,B2_J,dat2)
    Krow3,Kcol3,Kdat3 = assemble_global_CSR(A3_I,A3_J,B3_I,B3_J,dat3)

    Krow=Krow.astype(np.int32)
    Kcol=Kcol.astype(np.int32)
    Krow2=Krow2.astype(np.int32)
    Kcol2=Kcol2.astype(np.int32)
    Krow3=Krow3.astype(np.int32)
    Kcol3=Kcol3.astype(np.int32)
    #challenge is we likely have 3 different sparsity patterns so how should we
    #add them all using scipy???
    K1 = sp.csr_matrix((Kdat+Kdat2, Kcol, Krow), shape=(A_local_size[0],A_global_size[1]))
    K2 = sp.csr_matrix((Kdat3, Kcol3, Krow3), shape=(A_local_size[0],A_global_size[1]))
    
    ###############SUPG####################
    SUPG1_row,SUPG1_col,SUPG1_dat = assemble_global_CSR(SUPG1_I,SUPG1_J,SUPGB1_I,SUPGB1_J,SUPG_dat1)
    SUPG2_row,SUPG2_col,SUPG2_dat = assemble_global_CSR(SUPG2_I,SUPG2_J,SUPGB2_I,SUPGB2_J,SUPG_dat2)
    SUPG3_row,SUPG3_col,SUPG3_dat = assemble_global_CSR(SUPG3_I,SUPG3_J,SUPGB3_I,SUPGB3_J,SUPG_dat3)
    SUPG4_row,SUPG4_col,SUPG4_dat = assemble_global_CSR(SUPG4_I,SUPG4_J,SUPGB4_I,SUPGB4_J,SUPG_dat4)
   
    SUPG1_row = SUPG1_row.astype(np.int32)
    SUPG1_col = SUPG1_col.astype(np.int32)
    SUPG2_row = SUPG2_row.astype(np.int32)
    SUPG2_col = SUPG2_col.astype(np.int32)
    SUPG3_row = SUPG3_row.astype(np.int32)
    SUPG3_col = SUPG3_col.astype(np.int32)
    SUPG4_row = SUPG4_row.astype(np.int32)
    SUPG4_col = SUPG4_col.astype(np.int32)

    #assuming sparsity patterns are all identical
    SUPG1 = sp.csr_matrix((SUPG1_dat+SUPG2_dat+SUPG3_dat+SUPG4_dat,SUPG1_col,SUPG1_row), shape=(A_local_size[0],A_global_size[1]))
    #add the sparse matrices
    K = dt*(-K1+K2+SUPG1)
    
    #only works if there is boundary of domain 1 on this process
    if len(vals4[:,0]) != 0:
        K4 = PETScMatrix(MPI.COMM_SELF)
        fy.vector()[:] = np.array(vals4[0,:])
        K24 = u2*v2*fy*dx
        assemble(K24,tensor=K4)
        B4_I,B4_J,temp4 = K4.mat().getValuesCSR()
        blen4 = len(temp4)
        dat4 = np.zeros((blen4,len4))
        dat4[:,0] = temp4

        #K4 is the boundary integral dOmega1 x Omega2
        for i in range(1,len4):
            K4 = PETScMatrix(MPI.COMM_SELF)
            fy.vector()[:] = np.array(vals4[i,:])
            K24 = u2*v2*fy*dx
            assemble(K24,tensor=K4)

            _,_,temp4 = K4.mat().getValuesCSR()
            dat4[:,i] = temp4
    
    
        Krow4,Kcol4,Kdat4 = assemble_global_CSR(A4_I,A4_J,B4_I,B4_J,dat4)
   
        Krow4=Krow4.astype(np.int32)
        Kcol4=Kcol4.astype(np.int32)

        K3 = sp.csr_matrix((Kdat4, Kcol4, Krow4), shape=(A_local_size[0],A_global_size[1]))

        K = K + dt*K3
    PETSc.Sys.Print('Completed phase 1 of loading K')
    #############SUPG##########################
    #integrals that go 2 then 1
    
    cx_func = Function(V2)
    cy_func = Function(V2)
    csig_func = Function(V2)
    cthet_func = Function(V2)
    tau2 = Function(V2)

    #need value at a specific dof_coordinate in first domain
    cx_func.vector()[:] = np.array(c[0:N_dof_2,0])
    cy_func.vector()[:] = np.array(c[0:N_dof_2,1])
    csig_func.vector()[:] = np.array(c[0:N_dof_2,2])
    cthet_func.vector()[:] = np.array(c[0:N_dof_2,3])
    c1 = as_vector((cx_func,cy_func))
    c2 = as_vector((csig_func,cthet_func))
    tau2 = 1/(cx_func**2 + cy_func**2 + csig_func**2 + cthet_func**2)**0.5
    PETSc.Sys.Print('Still good up to here')
    SUPGB1 = tau2*csig_func*u2*v2*nabla_div(c2)*dx
    SUPGB2 = tau2*cthet_func*u2*v2*nabla_div(c2)*dx
    SUPGB3 = tau2*dot(c2,grad(u2))*dot(c2,grad(v2))*dx
    SUPGB4 = tau2*u2*nabla_div(c2)*dot(c2,grad(v2))*dx
    PETSc.Sys.Print('About to make some tensors') 
    SUPG_B1 = PETScMatrix(MPI.COMM_SELF)
    SUPG_B2 = PETScMatrix(MPI.COMM_SELF)
    SUPG_B3 = PETScMatrix(MPI.COMM_SELF)
    SUPG_B4 = PETScMatrix(MPI.COMM_SELF)
    
    PETSc.Sys.Print('Created tensors, going to load them now')
    
    assemble(SUPGB1,tensor=SUPG_B1)
    PETSc.Sys.Print('Matrix 1 assembled')
    assemble(SUPGB2,tensor=SUPG_B2)
    PETSc.Sys.Print('Matrix 2 assembled')
    assemble(SUPGB3,tensor=SUPG_B3)
    PETSc.Sys.Print('Matrix 3 assembled')
    assemble(SUPGB4,tensor=SUPG_B4)
    PETSc.Sys.Print('Matrix 4 assembled')
    
    PETSc.Sys.Print('About to save contents in CSR format')
    SUPGB1_I,SUPGB1_J,SUPGB1_temp = SUPG_B1.mat().getValuesCSR()
    SUPGB2_I,SUPGB2_J,SUPGB2_temp = SUPG_B2.mat().getValuesCSR()
    SUPGB3_I,SUPGB3_J,SUPGB3_temp = SUPG_B3.mat().getValuesCSR()
    SUPGB4_I,SUPGB4_J,SUPGB4_temp = SUPG_B4.mat().getValuesCSR()
    PETSc.Sys.Print('About to create array to store vals')
    SUPGB1_len = len(SUPGB1_temp)
    SUPGB2_len = len(SUPGB2_temp)
    SUPGB3_len = len(SUPGB3_temp)
    SUPGB4_len = len(SUPGB4_temp)

    SUPGB1_vals = np.zeros((SUPGB1_len,N_dof_1))
    SUPGB2_vals = np.zeros((SUPGB2_len,N_dof_1))
    SUPGB3_vals = np.zeros((SUPGB3_len,N_dof_1))
    SUPGB4_vals = np.zeros((SUPGB4_len,N_dof_1))
    PETSc.Sys.Print('Storing now, zeros arrays made')
    SUPGB1_vals[:,0] = SUPGB1_temp
    SUPGB2_vals[:,0] = SUPGB2_temp
    SUPGB3_vals[:,0] = SUPGB3_temp
    SUPGB4_vals[:,0] = SUPGB4_temp

    PETSc.Sys.Print('Entering loop 1 of phase 2')
    for b in range(1,N_dof_1):
        cx_func.vector()[:] = np.array(c[N_dof_2*b:N_dof_2*(b+1),0])
        cy_func.vector()[:] = np.array(c[N_dof_2*b:N_dof_2*(b+1),1])
        csig_func.vector()[:] = np.array(c[N_dof_2*b:N_dof_2*(b+1),2])
        cthet_func.vector()[:] = np.array(c[N_dof_2*b:N_dof_2*(b+1),3])
        c1 = as_vector((cx_func,cy_func))
        c2 = as_vector((csig_func,cthet_func))
        tau2 = 1/(cx_func**2 + cy_func**2 + csig_func**2 + cthet_func**2)**0.5
        SUPGB1 = tau2*csig_func*u2*v2*nabla_div(c2)*dx
        SUPGB2 = tau2*cthet_func*u2*v2*nabla_div(c2)*dx
        SUPGB3 = tau2*dot(c2,grad(u2))*dot(c2,grad(v2))*dx
        SUPGB4 = tau2*u2*nabla_div(c2)*dot(c2,grad(v2))*dx
    
        SUPG_B1 = PETScMatrix(MPI.COMM_SELF)
        SUPG_B2 = PETScMatrix(MPI.COMM_SELF)
        SUPG_B3 = PETScMatrix(MPI.COMM_SELF)
        SUPG_B4 = PETScMatrix(MPI.COMM_SELF)

        assemble(SUPGB1,tensor=SUPG_B1)
        assemble(SUPGB2,tensor=SUPG_B2)
        assemble(SUPGB3,tensor=SUPG_B3)
        assemble(SUPGB4,tensor=SUPG_B4)

        _,_,SUPGB1_temp = SUPG_B1.mat().getValuesCSR()
        _,_,SUPGB2_temp = SUPG_B2.mat().getValuesCSR()
        _,_,SUPGB3_temp = SUPG_B3.mat().getValuesCSR()
        _,_,SUPGB4_temp = SUPG_B4.mat().getValuesCSR()


        SUPGB1_vals[:,b] = SUPGB1_temp
        SUPGB2_vals[:,b] = SUPGB2_temp
        SUPGB3_vals[:,b] = SUPGB3_temp
        SUPGB4_vals[:,b] = SUPGB4_temp

    PETSc.Sys.Print('Completed 1/2 of phase 2')
    #need to move tau into 2 separate parts ????
    #or go back and do it right
    #########MARK Stopped here#################

    fx = Function(V1)
    fx2 = Function(V1)
    fx.vector()[:] = np.array(SUPGB1_vals[0,:])
    fx2.vector()[:] = np.array(SUPGB2_vals[0,:])
    SUPG1 = PETScMatrix()
    SUPG_1 = R*u1*dot(grad(v1),as_vector((fx,fx2)))*dx
    assemble(SUPG_1,tensor=SUPG1)


    fx.vector()[:] = np.array(SUPGB3_vals[0,:])
    fx2.vector()[:] = np.array(SUPGB4_vals[0,:])
    SUPG2 = PETScMatrix()
    SUPG_2 = R*u1*v1*(fx+fx2)*dx
    assemble(SUPG_2,tensor=SUPG2)


    SUPGA1_I,SUPGA1_J,SUPGA1_temp = SUPG1.mat().getValuesCSR()
    SUPGA2_I,SUPGA2_J,SUPGA2_temp = SUPG2.mat().getValuesCSR()

    alen1 = len(SUPGA1_temp)
    alen2 = len(SUPGA2_temp)

    dat1 = np.zeros((alen1,SUPGB1_len))
    dat2 = np.zeros((alen2,SUPGB2_len))
    
    dat1[:,0] = SUPGA1_temp
    dat2[:,0] = SUPGA2_temp
    PETSc.Sys.Print('About to enter final loop, completed pre-loop')

    for i in range(1,SUPGB1_len):

    
        fx.vector()[:] = np.array(SUPGB1_vals[i,:])
        fx2.vector()[:] = np.array(SUPGB2_vals[i,:])
        SUPG1 = PETScMatrix()
        SUPG_1 = R*u1*dot(grad(v1),as_vector((fx,fx2)))*dx
        assemble(SUPG_1,tensor=SUPG1)


        fx.vector()[:] = np.array(SUPGB3_vals[i,:])
        fx2.vector()[:] = np.array(SUPGB4_vals[i,:])
        SUPG2 = PETScMatrix()
        SUPG_2 = R*u1*v1*(fx+fx2)*dx
        assemble(SUPG_2,tensor=SUPG2)


        _,_,SUPGA1_temp = SUPG1.mat().getValuesCSR()
        _,_,SUPGA2_temp = SUPG2.mat().getValuesCSR()

    
        dat1[:,i] = SUPGA1_temp
        dat2[:,i] = SUPGA2_temp



    SUPG1_row,SUPG1_col,SUPG1_dat = assemble_global_CSR(SUPGA1_I,SUPGA1_J,SUPGB1_I,SUPGB1_J,np.transpose(dat1))
    SUPG2_row,SUPG2_col,SUPG2_dat = assemble_global_CSR(SUPGA2_I,SUPGA2_J,SUPGB2_I,SUPGB2_J,np.transpose(dat2))
    SUPG1_row = SUPG1_row.astype(np.int32)
    SUPG1_col = SUPG1_col.astype(np.int32)
    SUPG2_row = SUPG2_row.astype(np.int32)
    SUPG1 = sp.csr_matrix((SUPG1_dat+SUPG2_dat,SUPG1_col,SUPG1_row), shape=(A_local_size[0],A_global_size[1]))
    K=K+dt*SUPG1
    ##########################################
    #assign values to PETSc matrix
    A.setValuesCSR(K.indptr,K.indices,K.data)
    A.assemble()
    print('Finished assembly')
    return 0


def build_cartesian_mass_matrix_SUPG(mesh1,V1,mesh2,V2,c,N_dof_1,N_dof_2,dt,A):
        
    # mesh1 is geographic mesh
    # V1 is FunctionSpace for geographic mesh
    # mesh2 is spectral mesh
    # V2 is FunctionSpace for spectral mesh
    # c is group velocity vector at each d.o.f: should be #dof x 4 np vector
    # N_dof_2 is number of degrees of freedom in spectral domain
    # A is PETSc Matrix which is already preallocated, where output will be loaded

    n1 = FacetNormal(mesh1)
    n2 = FacetNormal(mesh2)
    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)
    u2 = TrialFunction(V2)
    v2 = TestFunction(V2)
    cx_func = Function(V1)
    cy_func = Function(V1)
    csig_func = Function(V1)
    cthet_func = Function(V1)

    #loop through dof_2 and get a N_dof_1xN_dof_1 sparse matrix
    #each matrix will have same sparsity pattern so get first one then
    #create numpy to store vals
    A_global_size = A.getSize()
    A_local_size = A.getLocalSize()

    #need value at a specific dof_coordinate in second domain
    cx_func.vector()[:] = np.array(c[0::N_dof_2,0])
    cy_func.vector()[:] = np.array(c[0::N_dof_2,1])
    csig_func.vector()[:] = np.array(c[0::N_dof_2,2])
    cthet_func.vector()[:] = np.array(c[0::N_dof_2,3])


    #R = (Circumradius(mesh1)**2+mesh2.hmax()**2)**0.5
    R =  (mesh1.hmax()**2 + mesh2.hmax()**2)**0.5
    tau = R/(cx_func**2 + cy_func**2 + csig_func**2 + cthet_func**2)**0.5
    c1 = as_vector((cx_func,cy_func))
    c2 = as_vector((csig_func,cthet_func))

    SUPG1 = tau*dot(c1,grad(v1))*u1*dx
    SUPG2 = tau*csig_func*u1*v1*dx
    SUPG3 = tau*cthet_func*u1*v1*dx


    SUPG_1 = PETScMatrix()
    SUPG_2 = PETScMatrix()
    SUPG_3 = PETScMatrix()
    assemble(SUPG1,tensor=SUPG_1)
    assemble(SUPG2,tensor=SUPG_2)
    assemble(SUPG3,tensor=SUPG_3)

    SUPG1_I,SUPG1_J,SUPG1_temp = SUPG_1.mat().getValuesCSR()
    SUPG2_I,SUPG2_J,SUPG2_temp = SUPG_2.mat().getValuesCSR()
    SUPG3_I,SUPG3_J,SUPG3_temp = SUPG_3.mat().getValuesCSR()


    SUPG1_len = len(SUPG1_temp)
    SUPG2_len = len(SUPG2_temp)
    SUPG3_len = len(SUPG3_temp)

    SUPG1_vals = np.zeros((SUPG1_len,N_dof_2))
    SUPG2_vals = np.zeros((SUPG2_len,N_dof_2))
    SUPG3_vals = np.zeros((SUPG3_len,N_dof_2))

    SUPG1_vals[:,0] = SUPG1_temp
    SUPG2_vals[:,0] = SUPG2_temp
    SUPG3_vals[:,0] = SUPG3_temp


    for a in range(1,N_dof_2):
        cx_func.vector()[:] = np.array(c[a::N_dof_2,0])
        cy_func.vector()[:] = np.array(c[a::N_dof_2,1])
        csig_func.vector()[:] = np.array(c[a::N_dof_2,2])
        cthet_func.vector()[:] = np.array(c[a::N_dof_2,3])
        c1 = as_vector((cx_func,cy_func))
        c2 = as_vector((csig_func,cthet_func))

        tau = R/(cx_func**2 + cy_func**2 + csig_func**2 + cthet_func**2)**0.5
        SUPG1 = tau*dot(c1,grad(v1))*u1*dx
        SUPG2 = tau*csig_func*u1*v1*dx
        SUPG3 = tau*cthet_func*u1*v1*dx
        SUPG_1 = PETScMatrix()
        SUPG_2 = PETScMatrix()
        SUPG_3 = PETScMatrix()
        assemble(SUPG1,tensor=SUPG_1)
        assemble(SUPG2,tensor=SUPG_2)
        assemble(SUPG3,tensor=SUPG_3)

        _,_,SUPG1_temp = SUPG_1.mat().getValuesCSR()
        _,_,SUPG2_temp = SUPG_2.mat().getValuesCSR()
        _,_,SUPG3_temp = SUPG_3.mat().getValuesCSR()
    
    
        SUPG1_vals[:,a] = SUPG1_temp
        SUPG2_vals[:,a] = SUPG2_temp
        SUPG3_vals[:,a] = SUPG3_temp


    fy = Function(V2)
    fy2 = Function(V2)

    SUPG_B1 = PETScMatrix(MPI.COMM_SELF)
    fy.vector()[:] = np.array(SUPG1_vals[0,:])
    SUPGB1 = u2*v2*(fy)*dx
    assemble(SUPGB1,tensor=SUPG_B1)


    SUPG_B2 = PETScMatrix(MPI.COMM_SELF)
    fy.vector()[:] = np.array(SUPG2_vals[0,:])
    fy2.vector()[:] = np.array(SUPG3_vals[0,:])
    SUPGB2 = u2*dot(grad(v2),as_vector((fy,fy2)))*dx
    assemble(SUPGB2,tensor=SUPG_B2)

    SUPGB1_I, SUPGB1_J, SUPGB1_temp = SUPG_B1.mat().getValuesCSR()
    SUPGB2_I, SUPGB2_J, SUPGB2_temp = SUPG_B2.mat().getValuesCSR()


    SUPGB1_len = len(SUPGB1_temp)
    SUPGB2_len = len(SUPGB2_temp)

    SUPG_dat1 = np.zeros((SUPGB1_len,SUPG1_len))
    SUPG_dat2 = np.zeros((SUPGB2_len,SUPG2_len))

    SUPG_dat1[:,0] = SUPGB1_temp
    SUPG_dat2[:,0] = SUPGB2_temp


    for i in range(1,SUPG1_len):


        SUPG_B1 = PETScMatrix(MPI.COMM_SELF)
        fy.vector()[:] = np.array(SUPG1_vals[i,:])
        SUPGB1 = u2*v2*(fy)*dx
        assemble(SUPGB1,tensor=SUPG_B1)


        SUPG_B2 = PETScMatrix(MPI.COMM_SELF)
        fy.vector()[:] = np.array(SUPG2_vals[i,:])
        fy2.vector()[:] = np.array(SUPG3_vals[i,:])
        SUPGB2 = u2*dot(grad(v2),as_vector((fy,fy2)))*dx
        assemble(SUPGB2,tensor=SUPG_B2)


        _,_,SUPGB1_temp = SUPG_B1.mat().getValuesCSR()
        _,_,SUPGB2_temp = SUPG_B2.mat().getValuesCSR()


        SUPG_dat1[:,i] = SUPGB1_temp
        SUPG_dat2[:,i] = SUPGB2_temp


    SUPG1_row,SUPG1_col,SUPG1_dat = assemble_global_CSR(SUPG1_I,SUPG1_J,SUPGB1_I,SUPGB1_J,SUPG_dat1)
    SUPG2_row,SUPG2_col,SUPG2_dat = assemble_global_CSR(SUPG2_I,SUPG2_J,SUPGB2_I,SUPGB2_J,SUPG_dat2)
   
    SUPG1_row = SUPG1_row.astype(np.int32)
    SUPG1_col = SUPG1_col.astype(np.int32)
    SUPG2_row = SUPG2_row.astype(np.int32)
    SUPG2_col = SUPG2_col.astype(np.int32)

    #assuming sparsity patterns are all identical
    SUPG1 = sp.csr_matrix((SUPG1_dat+SUPG2_dat,SUPG1_col,SUPG1_row), shape=(A_local_size[0],A_global_size[1]))
    #add the sparse matrices
    
    A.setValuesCSR(SUPG1.indptr,SUPG1.indices,SUPG1.data)
    A.assemble()
    return 0

def only_SUPG_terms(mesh1,V1,mesh2,V2,c,N_dof_1,N_dof_2,dt,A):
    #this function takes in empty PETsC matrix A and loads
    #all SUPG upwind terms =

    #first do terms by integrating domain 1 first then 2
    u1 = TrialFunction(V1)
    v1 = TestFunction(V1)
    u2 = TrialFunction(V2)
    v2 = TestFunction(V2)
    cx_func = Function(V1)
    cy_func = Function(V1)
    csig_func = Function(V1)
    cthet_func = Function(V1)

    #loop through dof_2 and get a N_dof_1xN_dof_1 sparse matrix
    #each matrix will have same sparsity pattern so get first one then
    #create numpy to store vals
    A_global_size = A.getSize()
    A_local_size = A.getLocalSize()
    
    #need value at a specific dof_coordinate in second domain
    cx_func.vector()[:] = np.array(c[0::N_dof_2,0])
    cy_func.vector()[:] = np.array(c[0::N_dof_2,1])
    csig_func.vector()[:] = np.array(c[0::N_dof_2,2])
    cthet_func.vector()[:] = np.array(c[0::N_dof_2,3])


    R =  (mesh1.hmax()**2 + mesh2.hmax()**2)**0.5
    #R = (Circumradius(mesh1)**2+mesh2.hmax()**2)**0.5
    #R =  mesh1.hmax()*mesh2.hmax()
    #R = (mesh1.hmax()**2 + mesh2.hmax()**2)**0.5
    #R =  (Circumradius(mesh1)**2 + mesh2.hmax()**2)**0.5
    tau = R/(cx_func**2 + cy_func**2 + csig_func**2 + cthet_func**2)**0.5
    c1 = as_vector((cx_func,cy_func))
    c2 = as_vector((csig_func,cthet_func))

    #series of weak forms

    #term1
    K1 = tau*dot(c1,grad(u1))*dot(c1,grad(v1))*dx
    
    #term2
    K2_a = tau*csig_func*dot(c1,grad(u1))*v1*dx
    K2_b = tau*cthet_func*dot(c1,grad(u1))*v1*dx
    
    #term3
    K3_a = tau*csig_func*u1*dot(c1,grad(v1))*dx
    K3_b = tau*cthet_func*u1*dot(c1,grad(v1))*dx
    
    #term5
    K5 = tau*nabla_div(c1)*u1*dot(c1,grad(v1))*dx
    
    #term6
    K6_a = tau*csig_func*nabla_div(c1)*u1*v1*dx
    K6_b = tau*cthet_func*nabla_div(c1)*u1*v1*dx

    #store in PETSc Matrices
    A1 = PETScMatrix()
    A2 = PETScMatrix()
    A3 = PETScMatrix()
    A4 = PETScMatrix()
    A5 = PETScMatrix()
    A6 = PETScMatrix()
    A7 = PETScMatrix()
    A8 = PETScMatrix()
    
    #load PETSc Matrices
    assemble(K1,tensor=A1)
    assemble(K2_a,tensor=A2)
    assemble(K2_b,tensor=A3)
    assemble(K3_a,tensor=A4)
    assemble(K3_b,tensor=A5)
    assemble(K5,tensor=A6)
    assemble(K6_a,tensor=A7)
    assemble(K6_b,tensor=A8)


    #get sizes of matrices (all should be identical or we will have issue)
    A1_I,A1_J,A1_temp = A1.mat().getValuesCSR()
    A2_I,A2_J,A2_temp = A2.mat().getValuesCSR()
    A3_I,A3_J,A3_temp = A3.mat().getValuesCSR()
    A4_I,A4_J,A4_temp = A4.mat().getValuesCSR()
    A5_I,A5_J,A5_temp = A5.mat().getValuesCSR()
    A6_I,A6_J,A6_temp = A6.mat().getValuesCSR()
    A7_I,A7_J,A7_temp = A7.mat().getValuesCSR()
    A8_I,A8_J,A8_temp = A8.mat().getValuesCSR()
    

    #create arrays to store functions

    A1_len = len(A1_temp)
    A2_len = len(A2_temp)
    A3_len = len(A3_temp)
    A4_len = len(A4_temp)
    A5_len = len(A5_temp)
    A6_len = len(A6_temp)
    A7_len = len(A7_temp)
    A8_len = len(A8_temp)

    A1_vals = np.zeros((A1_len,N_dof_2))
    A2_vals = np.zeros((A2_len,N_dof_2))
    A3_vals = np.zeros((A3_len,N_dof_2))
    A4_vals = np.zeros((A4_len,N_dof_2))
    A5_vals = np.zeros((A5_len,N_dof_2))
    A6_vals = np.zeros((A6_len,N_dof_2))
    A7_vals = np.zeros((A7_len,N_dof_2))
    A8_vals = np.zeros((A8_len,N_dof_2))

    A1_vals[:,0] = A1_temp
    A2_vals[:,0] = A2_temp
    A3_vals[:,0] = A3_temp
    A4_vals[:,0] = A4_temp
    A5_vals[:,0] = A5_temp
    A6_vals[:,0] = A6_temp
    A7_vals[:,0] = A7_temp
    A8_vals[:,0] = A8_temp


    #now loop thorugh N_dof_2 and store sparse matrices
    for a in range(1,N_dof_2):


        #need value at a specific dof_coordinate in second domain
        cx_func.vector()[:] = np.array(c[a::N_dof_2,0])
        cy_func.vector()[:] = np.array(c[a::N_dof_2,1])
        csig_func.vector()[:] = np.array(c[a::N_dof_2,2])
        cthet_func.vector()[:] = np.array(c[a::N_dof_2,3])


        tau = R/(cx_func**2 + cy_func**2 + csig_func**2 + cthet_func**2)**0.5
        c1 = as_vector((cx_func,cy_func))
        c2 = as_vector((csig_func,cthet_func))

        #series of weak forms
        #term1
        K1 = tau*dot(c1,grad(u1))*dot(c1,grad(v1))*dx
        #term2
        K2_a = tau*csig_func*dot(c1,grad(u1))*v1*dx
        K2_b = tau*cthet_func*dot(c1,grad(u1))*v1*dx
        #term3
        K3_a = tau*csig_func*u1*dot(c1,grad(v1))*dx
        K3_b = tau*cthet_func*u1*dot(c1,grad(v1))*dx
        #term5
        K5 = tau*nabla_div(c1)*u1*dot(c1,grad(v1))*dx
        #term6
        K6_a = tau*csig_func*nabla_div(c1)*u1*v1*dx
        K6_b = tau*cthet_func*nabla_div(c1)*u1*v1*dx

        #store in PETSc Matrices
        A1 = PETScMatrix()
        A2 = PETScMatrix()
        A3 = PETScMatrix()
        A4 = PETScMatrix()
        A5 = PETScMatrix()
        A6 = PETScMatrix()
        A7 = PETScMatrix()
        A8 = PETScMatrix()
    
        #load PETSc Matrices
        assemble(K1,tensor=A1)
        assemble(K2_a,tensor=A2)
        assemble(K2_b,tensor=A3)
        assemble(K3_a,tensor=A4)
        assemble(K3_b,tensor=A5)
        assemble(K5,tensor=A6)
        assemble(K6_a,tensor=A7)
        assemble(K6_b,tensor=A8)


        #get sizes of matrices (all should be identical or we will have issue)
        _,_,A1_temp = A1.mat().getValuesCSR()
        _,_,A2_temp = A2.mat().getValuesCSR()
        _,_,A3_temp = A3.mat().getValuesCSR()
        _,_,A4_temp = A4.mat().getValuesCSR()
        _,_,A5_temp = A5.mat().getValuesCSR()
        _,_,A6_temp = A6.mat().getValuesCSR()
        _,_,A7_temp = A7.mat().getValuesCSR()
        _,_,A8_temp = A8.mat().getValuesCSR()
    


        A1_vals[:,a] = A1_temp
        A2_vals[:,a] = A2_temp
        A3_vals[:,a] = A3_temp
        A4_vals[:,a] = A4_temp
        A5_vals[:,a] = A5_temp
        A6_vals[:,a] = A6_temp
        A7_vals[:,a] = A7_temp
        A8_vals[:,a] = A8_temp


    #now take sparse matrices and integrate in 2nd domain
    
    fy1 = Function(V2)
    fy2 = Function(V2)
    fy3 = Function(V2)
    fy4 = Function(V2)
    fy5 = Function(V2)
    fy6 = Function(V2)
    fy7 = Function(V2)
    fy8 = Function(V2)

    
    #term1
    fy1.vector()[:] = np.array(A1_vals[0,:])
    #term2
    fy2.vector()[:] = np.array(A2_vals[0,:])
    #term2
    fy3.vector()[:] = np.array(A3_vals[0,:])
    #term3
    fy4.vector()[:] = np.array(A4_vals[0,:])
    #term3
    fy5.vector()[:] = np.array(A5_vals[0,:])
    #term5
    fy6.vector()[:] = np.array(A6_vals[0,:])
    #term6
    fy7.vector()[:] = np.array(A7_vals[0,:])
    #term6
    fy8.vector()[:] = np.array(A8_vals[0,:])
    

    #assemble weak form
    L1 = u2*v2*(fy1)*dx
    L1 += u2*dot(grad(v2),as_vector((fy2,fy3)))*dx
    L1 += v2*dot(grad(u2),as_vector((fy4,fy5)))*dx
    L1 += u2*v2*fy6*dx
    L1 += u2*dot(grad(v2),as_vector((fy7,fy8)))*dx
    
    
    B1 = PETScMatrix(MPI.COMM_SELF)
    assemble(L1,tensor=B1)


    #now allocate arrays to stor data
    B1_I, B1_J, B1_temp = B1.mat().getValuesCSR()
    B1_len = len(B1_temp)
    dat1 = np.zeros((B1_len,A1_len))
    dat1[:,0] = B1_temp

    #now loop through each sparse entry from domain 1 and store data
    for b in range(1,A1_len):

        #term1
        fy1.vector()[:] = np.array(A1_vals[b,:])
        #term2
        fy2.vector()[:] = np.array(A2_vals[b,:])
        #term2
        fy3.vector()[:] = np.array(A3_vals[b,:])
        #term3
        fy4.vector()[:] = np.array(A4_vals[b,:])
        #term3
        fy5.vector()[:] = np.array(A5_vals[b,:])
        #term5
        fy6.vector()[:] = np.array(A6_vals[b,:])
        #term6
        fy7.vector()[:] = np.array(A7_vals[b,:])
        #term6
        fy8.vector()[:] = np.array(A8_vals[b,:])
    

        #assemble weak form
        L1 = u2*v2*(fy1)*dx
        L1 += u2*dot(grad(v2),as_vector((fy2,fy3)))*dx
        L1 += v2*dot(grad(u2),as_vector((fy4,fy5)))*dx
        L1 += u2*v2*fy6*dx
        L1 += u2*dot(grad(v2),as_vector((fy7,fy8)))*dx
    
    
        B1 = PETScMatrix(MPI.COMM_SELF)
        assemble(L1,tensor=B1)


        _,_,B1_temp = B1.mat().getValuesCSR()
        dat1[:,b] = B1_temp


    #now take data and send to PETScMatrix
    B1_row,B1_col,B1_dat = assemble_global_CSR(A1_I,A1_J,B1_I,B1_J,dat1)
   
    B1_row = B1_row.astype(np.int32)
    B1_col = B1_col.astype(np.int32)
    
    #assuming sparsity patterns are all identical
    #SUPG1 = sp.csr_matrix((B1_dat+B4_dat,B1_col,B1_row), shape=(A_local_size[0],A_global_size[1]))
    SUPG1 = sp.csr_matrix((B1_dat,B1_col,B1_row), shape=(A_local_size[0],A_global_size[1]))
   

    ############################################################
    ###########################################################
    #now do terms that require integration by second domain first
    cx_func = Function(V2)
    cy_func = Function(V2)
    csig_func = Function(V2)
    cthet_func = Function(V2)
    tau = Function(V2)

    #need value at a specific dof_coordinate in first domain
    cx_func.vector()[:] = np.array(c[0:N_dof_2,0])
    cy_func.vector()[:] = np.array(c[0:N_dof_2,1])
    csig_func.vector()[:] = np.array(c[0:N_dof_2,2])
    cthet_func.vector()[:] = np.array(c[0:N_dof_2,3])
    c1 = as_vector((cx_func,cy_func))
    c2 = as_vector((csig_func,cthet_func))
    #for now will only work for uniform grids!!!!
    
    #R =  Circumradius(mesh1)*mesh2.hmax()
    #R =  mesh1.hmax()*mesh2.hmax()
    #R = (mesh1.hmax()**2 + mesh2.hmax()**2)**0.5
    tau = 1/(cx_func**2 + cy_func**2 + csig_func**2 + cthet_func**2)**0.5
    
    #initiate weak forms
    #term4
    L1 = tau*dot(c2,grad(u2))*dot(c2,grad(v2))*dx
    
    #term7
    L2_a = tau*cx_func*nabla_div(c2)*u2*v2*dx
    L2_b = tau*cy_func*nabla_div(c2)*u2*v2*dx
    
    #term8
    L3 = tau*dot(c2,grad(v2))*u2*nabla_div(c2)*dx


    #now assemble stiffness matrices
    B1 = PETScMatrix(MPI.COMM_SELF)
    B2 = PETScMatrix(MPI.COMM_SELF)
    B3 = PETScMatrix(MPI.COMM_SELF)
    B4 = PETScMatrix(MPI.COMM_SELF)
    
    assemble(L1,tensor=B1)
    assemble(L2_a,tensor=B2)
    assemble(L2_b,tensor=B3)
    assemble(L3,tensor=B4)

    #unpack stiffness matrices
    
    B1_I, B1_J, B1_temp = B1.mat().getValuesCSR()
    B2_I, B2_J, B2_temp = B2.mat().getValuesCSR()
    B3_I, B3_J, B3_temp = B3.mat().getValuesCSR()
    B4_I, B4_J, B4_temp = B4.mat().getValuesCSR()

    B1_len = len(B1_temp)
    B2_len = len(B2_temp)
    B3_len = len(B3_temp)
    B4_len = len(B4_temp)

    B1_dat = np.zeros((B1_len,N_dof_1))
    B2_dat = np.zeros((B2_len,N_dof_1))
    B3_dat = np.zeros((B3_len,N_dof_1))
    B4_dat = np.zeros((B4_len,N_dof_1))

    B1_dat[:,0] = B1_temp
    B2_dat[:,0] = B2_temp
    B3_dat[:,0] = B3_temp
    B4_dat[:,0] = B4_temp


    #now loop throught all dof_1 and save in dat
    for a in range(1,N_dof_1):

        #need value at a specific dof_coordinate in first domain
        cx_func.vector()[:] = np.array(c[N_dof_2*a:N_dof_2*(a+1),0])
        cy_func.vector()[:] = np.array(c[N_dof_2*a:N_dof_2*(a+1),1])
        csig_func.vector()[:] = np.array(c[N_dof_2*a:N_dof_2*(a+1),2])
        cthet_func.vector()[:] = np.array(c[N_dof_2*a:N_dof_2*(a+1),3])
        c1 = as_vector((cx_func,cy_func))
        c2 = as_vector((csig_func,cthet_func))
        #for now will only work for uniform grids!!!!
        tau = 1/(cx_func**2 + cy_func**2 + csig_func**2 + cthet_func**2)**0.5
    
        #weak forms
        #term4
        L1 = tau*dot(c2,grad(u2))*dot(c2,grad(v2))*dx
    
        #term7
        L2_a = tau*cx_func*nabla_div(c2)*u2*v2*dx
        L2_b = tau*cy_func*nabla_div(c2)*u2*v2*dx
    
        #term8
        L3 = tau*dot(c2,grad(v2))*u2*nabla_div(c2)*dx
        #now assemble stiffness matrices
        B1 = PETScMatrix(MPI.COMM_SELF)
        B2 = PETScMatrix(MPI.COMM_SELF)
        B3 = PETScMatrix(MPI.COMM_SELF)
        B4 = PETScMatrix(MPI.COMM_SELF)
    
        assemble(L1,tensor=B1)
        assemble(L2_a,tensor=B2)
        assemble(L2_b,tensor=B3)
        assemble(L3,tensor=B4)

        #unpack stiffness matrices
        _,_,B1_temp = B1.mat().getValuesCSR()
        _,_,B2_temp = B2.mat().getValuesCSR()
        _,_,B3_temp = B3.mat().getValuesCSR()
        _,_,B4_temp = B4.mat().getValuesCSR()


        B1_dat[:,a] = B1_temp
        B2_dat[:,a] = B2_temp
        B3_dat[:,a] = B3_temp
        B4_dat[:,a] = B4_temp

    #now take this data and integrate through domain 1
    fx1 = Function(V1)
    fx2 = Function(V1)
    fx3 = Function(V1)
    fx4 = Function(V1)
    #R =  Circumradius(mesh1)*mesh2.hmax()
    
    #term4
    fx1.vector()[:] = np.array(B1_dat[0,:])
    #term7
    fx2.vector()[:] = np.array(B2_dat[0,:])
    fx3.vector()[:] = np.array(B3_dat[0,:])
    #term8
    fx4.vector()[:] = np.array(B4_dat[0,:])


    K1 = R*u1*v1*fx1*dx
    K1 += R*u1*dot(grad(v1),as_vector((fx2,fx3)))*dx
    K1 += R*u1*v1*fx4*dx
    
    A1 = PETScMatrix()
    assemble(K1,tensor=A1)

    #establish data matrices
    A1_I, A1_J, A1_temp = A1.mat().getValuesCSR()

    A1_len = len(A1_temp)

    dat1 = np.zeros((B1_len,A1_len))

    dat1[0,:] = A1_temp

    #now loop through the sparse entries from domain 2
    for b in range(1,B1_len):

        #term4
        fx1.vector()[:] = np.array(B1_dat[b,:])
        #term7
        fx2.vector()[:] = np.array(B2_dat[b,:])
        fx3.vector()[:] = np.array(B3_dat[b,:])
        #term8
        fx4.vector()[:] = np.array(B4_dat[b,:])


        K1 = R*u1*v1*fx1*dx
        K1 += R*u1*dot(grad(v1),as_vector((fx2,fx3)))*dx
        K1 += R*u1*v1*fx4*dx
    
        A1 = PETScMatrix()
        assemble(K1,tensor=A1)

        #establish data matrices
        _,_, A1_temp = A1.mat().getValuesCSR()

        dat1[b,:] = A1_temp

    #turn into sparse matrices
    B1_row,B1_col,B1_dat = assemble_global_CSR(A1_I,A1_J,B1_I,B1_J,dat1)
   
    B1_row = B1_row.astype(np.int32)
    B1_col = B1_col.astype(np.int32)

    #assuming sparsity patterns are all identical
    SUPG2 = sp.csr_matrix((B1_dat,B1_col,B1_row), shape=(A_local_size[0],A_global_size[1]))
    
    
    SUPG = dt*(SUPG1+SUPG2) 
    #add the sparse matrices
    A.setValuesCSR(SUPG.indptr,SUPG.indices,SUPG.data)
    A.assemble()

    

    return 0
