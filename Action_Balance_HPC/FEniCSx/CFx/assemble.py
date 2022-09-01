import numpy as np
from petsc4py import PETSc
from scipy import sparse as sp
import ufl
from dolfinx import fem,cpp
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


def build_action_balance_stiffness(domain1,domain2,V1,V2,A):
    #generates an assembled PETSc matrix
    #which is the stiffness matrix for the action balance equation

    n1 = ufl.FacetNormal(domain1)
    n2 = ufl.FacetNormal(domain2)

    u1 = ufl.TrialFunction(V1)
    v1 = ufl.TestFunction(V1)
    u2 = ufl.TrialFunction(V2)
    v2 = ufl.TestFunction(V2)

    cx_func = fem.Function(V1)
    cy_func = fem.Function(V1)
    csig_func = fem.Function(V1)
    cthet_func = fem.Function(V1)


    #loop through dof_2 and get a N_dof_1xN_dof_1 sparse matrix
    #each matrix will have same sparsity pattern so get first one then
    #create numpy to store vals
    A_global_size = A.getSize()
    A_local_size = A.getLocalSize()
    local_range1 = V1.dofmap.index_map.local_range
    dofs1 = np.arange(*local_range1,dtype=np.int32)
    local_size2 = V2.dofmap.index_map.size_local
   
    #need value at a specific dof_coordinate in second domain
    cx_func.vector.setValues(dofs1, np.array(c[0::local_size2,0]))
    #also need to propagate ghost values
    cx_func.vector.ghostUpdate()

    #need value at a specific dof_coordinate in second domain
    cy_func.vector.setValues(dofs1, np.array(c[0::local_size2,1]))
    #also need to propagate ghost values
    cy_func.vector.ghostUpdate()

    #need value at a specific dof_coordinate in second domain
    csig_func.vector.setValues(dofs1, np.array(c[0::local_size2,2]))
    #also need to propagate ghost values
    csig_func.vector.ghostUpdate()

    #need value at a specific dof_coordinate in second domain
    cthet_func.vector.setValues(dofs1, np.array(c[0::local_size2,3]))
    #also need to propagate ghost values
    cthet_func.vector.ghostUpdate()


    #create expressions and assemble linear forms
    K1 = cx_func*u1*v1.dx(0)*ufl.dx + cy_func*u1*v1.dx(1)*ufl.dx
    K2 = csig_func*u1*v1*ufl.dx
    K3 = cthet_func*u1*v1*ufl.dx
    K4 = dot(as_vector((cx_func,cy_func)),n1)*u1*v1*ufl.ds

    jit_parameters = {"cffi_extra_compile_args": ["-Ofast", "-march=native"],
            "cffi_libraries": ["m"]}


    #turn expressions into forms
    form_K1 = fem.form(K1, jit_params=jit_parameters)
    form_K2 = fem.form(K2, jit_params=jit_parameters)
    form_K3 = fem.form(K3, jit_params=jit_parameters)
    form_K4 = fem.form(K4, jit_params=jit_parameters)

    #generate sparsity patterns just 1 time
    #ideally they are all the same (except the boundary boi)
    sp1 = fem.create_sparsity_pattern(form_K1)
    sp2 = fem.create_sparsity_pattern(form_K2)
    sp3 = fem.create_sparsity_pattern(form_K3)
    sp4 = fem.create_sparsity_pattern(form_K4)


    #assemble each sparsity pattern
    sp1.assemble()
    sp2.assemble()
    sp3.assemble()
    sp4.assemble()

    print(f"Number of non-zeros in sparsity pattern: {sp.num_nonzeros}")
    print(f"Num dofs: {V.dofmap.index_map.size_local * V.dofmap.index_map_bs}")
    print(f"Entries per dof {sp.num_nonzeros/(V.dofmap.index_map.size_local * V.dofmap.index_map_bs)}")
    print(f"Dofs per cell {V.element.space_dimension}")
    print("-" * 25)

    #generate PETSc Matrices just one time
    A1 = cpp.la.petsc.create_matrix(domain1.comm, sp1)
    A2 = cpp.la.petsc.create_matrix(domain1.comm, sp2)
    A3 = cpp.la.petsc.create_matrix(domain1.comm, sp3)
    A4 = cpp.la.petsc.create_matrix(domain1.comm, sp4)

    #assemble the PETSc Matrices
    fem.petsc.assemble_matrix(A1, form_K1)
    fem.petsc.assemble_matrix(A2, form_K2)
    fem.petsc.assemble_matrix(A3, form_K3)
    fem.petsc.assemble_matrix(A4, form_K4)
    
    A1.assemble()
    A2.assemble()
    A3.assemble()
    A4.assemble()

    #store CSR entries in a large matrix
    #store sparsity pattern (rows,columns, vals)
    A1_I,A1_J,temp =  A1.mat().getValuesCSR()
    A2_I,A2_J,temp2 = A2.mat().getValuesCSR()
    A3_I,A3_J,temp3 = A3.mat().getValuesCSR()
    A4_I,A4_J,temp4 = A4.mat().getValuesCSR()
    len1 = len(temp)
    len2 = len(temp2)
    len3 = len(temp3)
    len4 = len(temp4)
    
    #create np to store N_dof_2 sets of vals
    vals1 = np.zeros((len1,local_size2))
    vals2 = np.zeros((len2,local_size2))
    vals3 = np.zeros((len3,local_size2))
    vals4 = np.zeros((len4,local_size2))
    vals1[:,0] = temp
    vals2[:,0] = temp2
    vals3[:,0] = temp3
    vals4[:,0] = temp4

    #for some reason it likes to add things
    #so until we figure it out we need to zero entries
    A1.zeroEntries()
    A2.zeroEntries()
    A3.zeroEntries()
    A4.zeroEntries()

    #iterate over each dof2
    for a in range(local_size2):

        #need value at a specific dof_coordinate in second domain
        cx_func.vector.setValues(dofs1, np.array(c[a::local_size2,0]))
        #also need to propagate ghost values
        cx_func.vector.ghostUpdate()

        #need value at a specific dof_coordinate in second domain
        cy_func.vector.setValues(dofs1, np.array(c[a::local_size2,1]))
        #also need to propagate ghost values
        cy_func.vector.ghostUpdate()

        #need value at a specific dof_coordinate in second domain
        csig_func.vector.setValues(dofs1, np.array(c[a::local_size2,2]))
        #also need to propagate ghost values
        csig_func.vector.ghostUpdate()

        #need value at a specific dof_coordinate in second domain
        cthet_func.vector.setValues(dofs1, np.array(c[a::local_size2,3]))
        #also need to propagate ghost values
        cthet_func.vector.ghostUpdate()

        fem.petsc.assemble_matrix(A1,form_K1)
        fem.petsc.assemble_matrix(A2, form_K2)
        fem.petsc.assemble_matrix(A3, form_K3)
        fem.petsc.assemble_matrix(A4, form_K4)
        A1.assemble()
        A2.assemble()
        A3.assemble()
        A4.assemble()

        #store CSR entries in a large matrix
        #store sparsity pattern (rows,columns, vals)
        _,_,temp =  A1.mat().getValuesCSR()
        _,_,temp2 = A2.mat().getValuesCSR()
        _,_,temp3 = A3.mat().getValuesCSR()
        _,_,temp4 = A4.mat().getValuesCSR()


        vals1[:,0] = temp
        vals2[:,0] = temp2
        vals3[:,0] = temp3
        vals4[:,0] = temp4


        #for some reason it likes to add things
        #so until we figure it out we need to zero entries
        A1.zeroEntries()
        A2.zeroEntries()
        A3.zeroEntries()
        A4.zeroEntries()


