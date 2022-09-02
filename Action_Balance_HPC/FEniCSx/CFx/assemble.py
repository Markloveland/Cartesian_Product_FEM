import numpy as np
from mpi4py import MPI
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
    M1_I,M1_J,M1_A = M1_pet.getValuesCSR()
    M1_NNZ = M1_I[1:]-M1_I[:-1]
    M2_I,M2_J,M2_A = M2_pet.getValuesCSR()
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


def build_action_balance_stiffness(domain1,domain2,V1,V2,c,dt,A):
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
    local_size1 = V1.dofmap.index_map.size_local
    
    local_range2 = V2.dofmap.index_map.local_range
    dofs2 = np.arange(*local_range2,dtype=np.int32)
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
    K4 = ufl.dot(ufl.as_vector((cx_func,cy_func)),n1)*u1*v1*ufl.ds

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

    #print(f"Number of non-zeros in sparsity pattern: {sp1.num_nonzeros}")
    #print(f"Num dofs: {V1.dofmap.index_map.size_local * V1.dofmap.index_map_bs}")
    #print(f"Entries per dof {sp1.num_nonzeros/(V1.dofmap.index_map.size_local * V1.dofmap.index_map_bs)}")
    #print(f"Dofs per cell {V1.element.space_dimension}")
    #print("-" * 25)

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
    A1_I,A1_J,temp =  A1.getValuesCSR()
    A2_I,A2_J,temp2 = A2.getValuesCSR()
    A3_I,A3_J,temp3 = A3.getValuesCSR()
    A4_I,A4_J,temp4 = A4.getValuesCSR()
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
    for a in range(1,local_size2):

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
        _,_,temp =  A1.getValuesCSR()
        _,_,temp2 = A2.getValuesCSR()
        _,_,temp3 = A3.getValuesCSR()
        _,_,temp4 = A4.getValuesCSR()


        vals1[:,a] = temp
        vals2[:,a] = temp2
        vals3[:,a] = temp3
        vals4[:,a] = temp4


        #for some reason it likes to add things
        #so until we figure it out we need to zero entries
        A1.zeroEntries()
        A2.zeroEntries()
        A3.zeroEntries()
        A4.zeroEntries()

    
    #now for each entry in sparse N_dof_1 x N_dof_1 matrix need to evaluate
    # int_Omega2 fy ... dy
    #like before, first need to get sparsity patterns
    fy1 = fem.Function(V2)
    fy2 = fem.Function(V2)
    fy3 = fem.Function(V2)

    #need value at a specific dof_coordinate in second domain
    #print(vals1.shape)
    fy1.vector.setValues(dofs2, np.array(vals1[0,:]))
    #also need to propagate ghost values
    fy1.vector.ghostUpdate()

    #need value at a specific dof_coordinate in second domain
    fy2.vector.setValues(dofs2, np.array(vals2[0,:]))
    #also need to propagate ghost values
    fy2.vector.ghostUpdate()
    
    #need value at a specific dof_coordinate in second domain
    fy3.vector.setValues(dofs2, np.array(vals3[0,:]))
    #also need to propagate ghost values
    fy3.vector.ghostUpdate()

    #assemble the weak form
    K21 = -u2*v2*fy1*ufl.dx
    K21 += -ufl.inner(u2*ufl.as_vector((fy2,fy3)),ufl.grad(v2))*ufl.dx
    K21 += u2*v2*ufl.dot(ufl.as_vector((fy2,fy3)),n2)*ufl.ds

    #turn expression into form
    form_K21 = fem.form(K21, jit_params=jit_parameters)

    #generate sparsity patterns just 1 time
    #ideally they are all the same (except the boundary boi)
    sp21 = fem.create_sparsity_pattern(form_K21)

    #assemble each sparsity pattern
    sp21.assemble()

    #generate PETSc Matrices just one time
    B1 = cpp.la.petsc.create_matrix(domain2.comm, sp21)

    #assemble the PETSc Matrices
    fem.petsc.assemble_matrix(B1, form_K21)
    
    B1.assemble()

    B1_I,B1_J,temp = B1.getValuesCSR()

    blen1 = len(temp)
    dat1 = np.zeros((blen1,len1))
    dat1[:,0] = temp
    #for some reason it likes to add things
    #so until we figure it out we need to zero entries
    B1.zeroEntries()

    #KEY! IDK IF TRUE BUT ASSUMING length of sparse matrixes K1,K2,K3 were same
    #If not, then will need separate loops

    for i in range(1,len1):

        #need value at a specific dof_coordinate in second domain
        fy1.vector.setValues(dofs2, np.array(vals1[i,:]))
        #also need to propagate ghost values
        fy1.vector.ghostUpdate()

        #need value at a specific dof_coordinate in second domain
        fy2.vector.setValues(dofs2, np.array(vals2[i,:]))
        #also need to propagate ghost values
        fy2.vector.ghostUpdate()
    
        #need value at a specific dof_coordinate in second domain
        fy3.vector.setValues(dofs2, np.array(vals3[i,:]))
        #also need to propagate ghost values
        fy3.vector.ghostUpdate()


        fem.petsc.assemble_matrix(B1,form_K21)
        B1.assemble()

        _,_,temp = B1.getValuesCSR()

        dat1[:,i] = temp

        B1.zeroEntries()

    Krow,Kcol,Kdat = assemble_global_CSR(A1_I,A1_J,B1_I,B1_J,dat1)

    Krow=Krow.astype(np.int32)
    Kcol=Kcol.astype(np.int32)
    #challenge is we likely have 3 different sparsity patterns so how should we
    #add them all using scipy???
    K = sp.csr_matrix((Kdat, Kcol, Krow), shape=(A_local_size[0],A_global_size[1]))
    #add the sparse matrices
    K = dt*K
    
    

    #now need to repeat last loop because sometimes we have empty global boundary
    if len(vals4[:,0]) != 0:
        #need value at a specific dof_coordinate in second domain
        fy1.vector.setValues(dofs2, np.array(vals4[0,:]))
        #also need to propagate ghost values
        fy1.vector.ghostUpdate()
        K22 = u2*v2*fy1*ufl.dx

        #turn expression into form
        form_K22 = fem.form(K22, jit_params=jit_parameters)

        #generate sparsity patterns just 1 time
        #ideally they are all the same (except the boundary boi)
        sp22 = fem.create_sparsity_pattern(form_K22)

        #assemble each sparsity pattern
        sp22.assemble()

        #generate PETSc Matrices just one time
        B2 = cpp.la.petsc.create_matrix(domain2.comm, sp22)

        #assemble the PETSc Matrices
        fem.petsc.assemble_matrix(B2, form_K22)
    
        B2.assemble()

        B2_I,B2_J,temp2 = B2.getValuesCSR()
        blen2 = len(temp2)
        dat2 = np.zeros((blen2,len4))
        dat2[:,0] = temp2

        B2.zeroEntries()


        #K4 is the boundary integral dOmega1 x Omega2
        for i in range(1,len4):
            #need value at a specific dof_coordinate in second domain
            fy1.vector.setValues(dofs2, np.array(vals4[i,:]))
            #also need to propagate ghost values
            fy1.vector.ghostUpdate()

            fem.petsc.assemble_matrix(B2,form_K22)
            B2.assemble()

            _,_,temp2 = B2.getValuesCSR()

            dat2[:,i] = temp2

            B2.zeroEntries()

        Krow2,Kcol2,Kdat2 = assemble_global_CSR(A4_I,A4_J,B2_I,B2_J,dat2)

        Krow2=Krow2.astype(np.int32)
        Kcol2=Kcol2.astype(np.int32)

        K2 = sp.csr_matrix((Kdat2, Kcol2, Krow2), shape=(A_local_size[0],A_global_size[1]))

        K = K + dt*K2

    #assign values to PETSc matrix
    A.setValuesCSR(K.indptr,K.indices,K.data)
    A.assemble()
    return 0
