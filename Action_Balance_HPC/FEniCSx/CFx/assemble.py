import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from scipy import sparse as sp
import ufl
from dolfinx import fem,cpp
import CFx.forms
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

    Krow=Krow.astype(np.int32)
    Kcol=Kcol.astype(np.int32)
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

def init_dat(domain1,K1,jit_parameters = {"cffi_extra_compile_args": ["-Ofast", "-march=native"],
    "cffi_libraries": ["m"]}):
    #give a ufl form, generate a sparsity pattern

    form_K1 = fem.form(K1, jit_params=jit_parameters)
    #generate sparsity patterns just 1 time
    #ideally they are all the same (except the boundary boi)
    sp1 = fem.create_sparsity_pattern(form_K1)

    #assemble each sparsity pattern
    sp1.assemble()

    #generate PETSc Matrices just one time
    A1 = cpp.la.petsc.create_matrix(domain1.comm, sp1)

    #assemble the PETSc Matrices
    fem.petsc.assemble_matrix(A1, form_K1)

    A1.assemble()

    return form_K1,A1

def init_array1(A1,local_size2):
    #needs petsc matrix
    #store CSR entries in a large matrix
    #store sparsity pattern (rows,columns, vals)
    A1_I,A1_J,temp =  A1.getValuesCSR()
    len1 = len(temp)

    #create np to store N_dof_2 sets of vals
    vals1 = np.zeros((len1,local_size2))
    vals1[:,0] = temp
    return A1_I,A1_J,vals1,len1

def init_array2(B1,len1):

    B1_I,B1_J,temp = B1.getValuesCSR()

    blen1 = len(temp)
    dat1 = np.zeros((blen1,len1))
    dat1[:,0] = temp
    return B1_I,B1_J,dat1

def update_c(c_func,c_vals,dofs1,local_size2,node_num,subdomain=0):
    if subdomain == 0:
        ctr = 0
        for c in c_func:
            #need value at a specific dof_coordinate in second domain
            c.vector.setValues(dofs1, np.array(c_vals[node_num::local_size2,ctr]))
            #also need to propagate ghost values
            c.vector.ghostUpdate()
            ctr = ctr+1
    if subdomain == 1:
        ctr = 0
        for c in c_func:
            c.vector.setValues(dofs1,np.array(c_vals[node_num*local_size2:(node_num+1)*local_size2,ctr]))
            c.vector.ghostUpdate()
            ctr = ctr+1
    return 0

def update_tau(tau,c_vals,mesh1,mesh2,local_size2,dofs,node_num,subdomain=0):
    #hardcoded for uniform mesh this time
    tdim = mesh1.topology.dim
    num_cells = mesh1.topology.index_map(tdim).size_local
    h1 = cpp.mesh.h(mesh1, tdim, range(num_cells)).max()
    
    tdim = mesh2.topology.dim
    num_cells = mesh2.topology.index_map(tdim).size_local
    h2 = cpp.mesh.h(mesh2, tdim, range(num_cells)).max()

    h = np.sqrt(h1**2+h2**2)
    if subdomain == 0:
        temp = c_vals[node_num::local_size2,:]**2
    if subdomain == 1:
        temp = c_vals[node_num*local_size2:(node_num+1)*local_size2,:]**2
    c_mag = np.sqrt(temp.sum(axis=1))
    #tau = h / /|c/|
    tau.vector.setValues(dofs,np.array(h/c_mag))
    tau.vector.ghostUpdate()
    return 0

def assemble_subdomain1(K_vec,domain1,local_size1,local_size2,c_func,c_vals,dofs1,j=0): 
    if j == 0:
        local_size = local_size2
    if j ==1:
        local_size = local_size1

    form_K = []
    A_vec = []

    for K in K_vec:
        form_K1,A1 = init_dat(domain1,K)
        form_K.append(form_K1)
        A_vec.append(A1)
        
    A_I = []
    A_J = []
    vals = []
    lens1 = []
    
    for A1 in A_vec:
        A1_I,A1_J,vals1,len1 = init_array1(A1,local_size)
        A_I.append(A1_I)
        A_J.append(A1_J)
        vals.append(vals1)
        lens1.append(len1)
        A1.zeroEntries()
    #print(f"Number of non-zeros in sparsity pattern: {sp1.num_nonzeros}")
    #print(f"Num dofs: {V1.dofmap.index_map.size_local * V1.dofmap.index_map_bs}")
    #print(f"Entries per dof {sp1.num_nonzeros/(V1.dofmap.index_map.size_local * V1.dofmap.index_map_bs)}")
    #print(f"Dofs per cell {V1.element.space_dimension}")
    #print("-" * 25)

    #iterate over each dof2
    for a in range(1,local_size):
        update_c(c_func,c_vals,dofs1,local_size2,a,subdomain=j)
        ctr = 0
        
        for K in form_K:
            fem.petsc.assemble_matrix(A_vec[ctr],K)
            A_vec[ctr].assemble()
            _,_,temp = A_vec[ctr].getValuesCSR()
            vals[ctr][:,a] = temp
            A_vec[ctr].zeroEntries()
            ctr=ctr+1

    
    return A_I,A_J,vals,lens1


def update_f(fy,dofs2,vals,entry_no):
    ctr = 0
    for fy1 in fy:
        fy1.vector.setValues(dofs2, np.array(vals[ctr][entry_no,:]))
        #also need to propagate ghost values
        fy1.vector.ghostUpdate()
        ctr=ctr+1
    return 0

def fetch_params(A,V1,V2):
    #get parameters related to matrix construction
    A_global_size = A.getSize()
    A_local_size = A.getLocalSize()
    
    local_range1 = V1.dofmap.index_map.local_range
    dofs1 = np.arange(*local_range1,dtype=np.int32)
    local_size1 = V1.dofmap.index_map.size_local
    
    local_range2 = V2.dofmap.index_map.local_range
    dofs2 = np.arange(*local_range2,dtype=np.int32)
    local_size2 = V2.dofmap.index_map.size_local

    return A_global_size,A_local_size,dofs1,local_size1,dofs2,local_size2

def assemble_subdomain2(domain2,K21,len1,fy,dofs2,vals):
    #need value at a specific dof_coordinate in second domain
    form_K21,B1 = init_dat(domain2,K21)
    B1_I,B1_J,dat1 = init_array2(B1,len1)
    #for some reason it likes to add things
    #so until we figure it out we need to zero entries
    B1.zeroEntries()

    #KEY! IDK IF TRUE BUT ASSUMING length of sparse matrixes K1,K2,K3 were same
    #If not, then will need separate loops

    for i in range(1,len1):

        update_f(fy,dofs2,vals,i)
        fem.petsc.assemble_matrix(B1,form_K21)
        B1.assemble()
        _,_,temp = B1.getValuesCSR()
        dat1[:,i] = temp
        B1.zeroEntries()

    return B1_I,B1_J,dat1


def build_action_balance_stiffness(domain1,domain2,V1,V2,c_vals,dt,A,method='CG'):
    #given a method, pulls in fenics weak form and
    #generates an assembled PETSc matrix
    #which is the stiffness matrix for the action balance equation

    #get parameters relating to stiffness matrix size
    A_global_size,A_local_size,dofs1,local_size1,dofs2,local_size2 = fetch_params(A,V1,V2)

    #first pull the proper weak form:
    c_func,K_vec,fy,K2,K_bound,fy4,K3 = CFx.forms.CG_weak_form(domain1,domain2,V1,V2)

    #if method == 'SUPG':
    #    c_func,K_vec,fy,K2,K_bound,fy4,K3,tau,tau2,c2,K2_vol,fx,K2_vol_x = CFx.forms.CG_weak_form(domain1,domain2,V1,V2,SUPG='on')
    #    #(still need to update tau)
    #    #update_c([tau] ...
    #    update_tau(tau,c_vals,domain1,domain2,local_size2,dofs1,0,subdomain=0)
    #    update_tau(tau2,c_vals,domain1,domain2,local_size2,dofs2,0,subdomain=1)
    
    #integrate volume terms in first subdomain
    update_c(c_func,c_vals,dofs1,local_size2,0)
    A_I,A_J,vals,lens1 = assemble_subdomain1(K_vec,domain1,local_size1,local_size2,c_func,c_vals,dofs1)
    #integrate resulting terms in second subdomain
    update_f(fy,dofs2,vals,0)
    B1_I,B1_J,dat1 = assemble_subdomain2(domain2,K2,lens1[0],fy,dofs2,vals)
    #formulate CSR format entries
    Krow,Kcol,Kdat = assemble_global_CSR(A_I[0],A_J[0],B1_I,B1_J,dat1)
    #use scipy make store temporary csr matrix
    K = sp.csr_matrix((Kdat, Kcol, Krow), shape=(A_local_size[0],A_global_size[1]))
    #add the sparse matrices
    K = dt*K

    #now integrate boundary terms in first subdomain
    update_c(c_func,c_vals,dofs1,local_size2,0)
    A_I,A_J,vals,lens1 = assemble_subdomain1(K_bound,domain1,local_size1,local_size2,c_func,c_vals,dofs1)
    #some partitions in the first subdomain dont contain any global boundary so check
    if len(vals[0][:,0]) != 0:
        #integrate resulting  terms in second subdomain
        update_f(fy4,dofs2,vals,0)
        B2_I,B2_J,dat2 = assemble_subdomain2(domain2,K3,lens1[0],fy4,dofs2,vals)
        #formulate CSR format entries
        Krow2,Kcol2,Kdat2 = assemble_global_CSR(A_I[0],A_J[0],B2_I,B2_J,dat2)
        K2 = sp.csr_matrix((Kdat2, Kcol2, Krow2), shape=(A_local_size[0],A_global_size[1]))
        #add scipy matrices
        K = K + dt*K2
    
    #if method is stabilized, will have terms that need to be integrated in first subdomain then second
    if method == 'SUPG':
        c_func,tau,K_vol,fy,K_vol_y,tau2,c2,K2_vol,fx,K2_vol_x = CFx.forms.SUPG_weak_form_standalone(domain1,domain2,V1,V2) 
        
        
        #integrate volume terms in first subdomain
        update_tau(tau,c_vals,domain1,domain2,local_size2,dofs1,0,subdomain=0)
        update_tau(tau2,c_vals,domain1,domain2,local_size2,dofs2,0,subdomain=1)
        
        update_c(c_func,c_vals,dofs1,local_size2,0)
        
        A_I,A_J,vals,lens1 = assemble_subdomain1(K_vol,domain1,local_size1,local_size2,c_func,c_vals,dofs1)
        #integrate resulting terms in second subdomain
        update_f(fy,dofs2,vals,0)
        B1_I,B1_J,dat1 = assemble_subdomain2(domain2,K_vol_y,lens1[0],fy,dofs2,vals)
        #formulate CSR format entries
        Krow,Kcol,Kdat = assemble_global_CSR(A_I[0],A_J[0],B1_I,B1_J,dat1)
        #use scipy make store temporary csr matrix
        K_SUPG = sp.csr_matrix((Kdat, Kcol, Krow), shape=(A_local_size[0],A_global_size[1]))
        #add the sparse matrices
        K = K + dt*K_SUPG
        
        
        #integrate volume terms in second subdomain
        update_c(c2,c_vals,dofs2,local_size2,0,subdomain=1)
        B_I,B_J,vals,lens1 = assemble_subdomain1(K2_vol,domain2,local_size1,local_size2,c2,c_vals,dofs2,j=1)
        update_f(fx,dofs1,vals,0)
        A1_I,A1_J,dat3 = assemble_subdomain2(domain1,K2_vol_x,lens1[0],fx,dofs1,vals)
        
        Krow3,Kcol3,Kdat3 = assemble_global_CSR(A1_I,A1_J,B_I[0],B_J[0],dat3.T)
        K3 = sp.csr_matrix((Kdat3, Kcol3, Krow3), shape=(A_local_size[0],A_global_size[1]))
        K = K + dt*K3
        
    #assign values to PETSc matrix
    A.setValuesCSR(K.indptr,K.indices,K.data)
    A.assemble()
    return 0

def build_RHS(domain1,domain2,V1,V2,c_vals,A):
    A_global_size,A_local_size,dofs1,local_size1,dofs2,local_size2 = fetch_params(A,V1,V2)
    L,g,L_y,tau,c_func = CFx.forms.SUPG_RHS(domain1,domain2,V1,V2)
    update_tau(tau,c_vals,domain1,domain2,local_size2,dofs1,0,subdomain=0)
    #integrate volume terms in first subdomain
    update_c(c_func,c_vals,dofs1,local_size2,0)
    A_I,A_J,vals,lens1 = assemble_subdomain1(L,domain1,local_size1,local_size2,c_func,c_vals,dofs1)
    #integrate resulting terms in second subdomain
    update_f(g,dofs2,vals,0)
    B1_I,B1_J,dat1 = assemble_subdomain2(domain2,L_y,lens1[0],g,dofs2,vals)
    #formulate CSR format entries
    Krow,Kcol,Kdat = assemble_global_CSR(A_I[0],A_J[0],B1_I,B1_J,dat1)
    #use scipy make store temporary csr matrix
    K = sp.csr_matrix((Kdat, Kcol, Krow), shape=(A_local_size[0],A_global_size[1]))

    A.setValuesCSR(K.indptr,K.indices,K.data)
    A.assemble()
    return 0
