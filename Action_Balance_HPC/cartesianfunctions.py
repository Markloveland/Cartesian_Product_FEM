import numpy as np
from fenics import *
import petsc4py
petsc4py.init()
from petsc4py import PETSc
from mpi4py import MPI
from scipy import sparse as sp

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
    global_boundary_dofs=np.unique(global_boundary_dofs,axis=0)

    #have cartesian product of dof at entire boundary (this form should be easy to get coordinates in if needed)
    #now need to convert to global system dof as the kron function does
    global_boundary_dofs=cartesian_form_to_kroneck_form(global_boundary_dofs, len(dof_coordinates2))
    global_boundary_dofs=global_boundary_dofs.astype("int32")
    return global_boundary_dofs

def assemble_global_CSR(Arow,Acol,Brow,Bcol,dat):
    #assembles inputs to load PETSc CSR matrix
    nnzA = Arow[1:] - Arow[:-1]
    nnzB = Brow[1:] - Brow[:-1]
    nA = len(nnzA)
    nB = len(nnzB)
    #print('Num A')
    #print(len(nnzA))
    #print('Num B')
    #print(len(nnzB))
    Kcol = np.zeros(dat.size)
    Krow = np.zeros(len(nnzA)*len(nnzB)+1)
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

