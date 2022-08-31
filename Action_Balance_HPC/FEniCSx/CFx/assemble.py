import numpy as np
from petsc4py import PETSc
from scipy import sparse as sp
#all scripts related to assembling the global PETSc matrices


def assemble_global_CSR(Arow,Acol,Brow,Bcol,dat):
    #assembles inputs to load PETSc CSR matrix
    #takes domain 2 matrices and putting in
    #block by block
    nnzA = Arow[1:] - Arow[:-1]
    nnzB = Brow[1:] - Brow[:-1]
    nA = len(nnzA)
    nB = len(nnzB)
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
