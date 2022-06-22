"""
Poisson with non-separable, varying diffusivity coefficient
This algorithm for loading will be same required of Action Balance Equation
    \/ . k \/u = f
only 2D case to start
"""

from __future__ import print_function
import numpy as np
from fenics import *
import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy import sparse as sp
import petsc4py
petsc4py.init()
from petsc4py import PETSc
import cartesianfunctions as CF

#get MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
# Create cartesian mesh of two intervals and define function spaces
nx = 200
ny = 10
#set initial time
t = 0
#set time step


####################################################################
#Subdomain 1
#the first subdomain will be split amongst processors
# this is set up ideally for subdomain 1 > subdomain 2
mesh1 = UnitIntervalMesh(comm,nx)
V1 = FunctionSpace(mesh1, 'P', 1)
u1 = TrialFunction(V1)
v1 = TestFunction(V1)
####################################################################
####################################################################
#Subdomain 2
#now we want entire second subdomain on EACH processor, so this will always be the smaller mesh
#MPI.COMM_SELF to not partition mesh
mesh2 = UnitIntervalMesh(MPI.COMM_SELF,ny)
V2 = FunctionSpace(mesh2, 'P', 1)
u2 = TrialFunction(V2)
v2 = TestFunction(V2)
###################################################################
###################################################################
#need local mass matrices to build global mass matrix
#mass of subdomain 1
m1 = u1*v1*dx
M1_pet = PETScMatrix()
assemble(m1,tensor=M1_pet)
m2 = u2*v2*dx
M2_pet = PETScMatrix(MPI.COMM_SELF)
assemble(m2,tensor=M2_pet)
###################################################################
###################################################################
#Allocate global mass matrix
#need to generate global mass matrix to get global matrix layout and sparsity patterns
#Assemble global matrices (product of each subdomain)
M1_sizes = M1_pet.mat().getLocalSize()
M1_global_size =M1_pet.mat().getSize() 
M2_sizes = M2_pet.mat().getLocalSize()

local_rows = int(M1_sizes[0]*M2_sizes[0])
global_rows = int(M1_global_size[0]*M2_sizes[0])
local_cols = int(M1_sizes[1]*M2_sizes[1])
global_cols = int(M1_global_size[1]*M2_sizes[1])

#first I need to create an mpi matrix of the appropriate size and start storing values
M = PETSc.Mat()
M.create(comm=comm)
M.setSizes(([local_rows,global_rows],[local_cols,global_cols]),bsize=1)
M.setFromOptions()
M.setType('aij')
M.setUp()
#also need global stiffness matrix
#same exact structure as M
A = M.duplicate()
#get ownership range
local_range = M.getOwnershipRange()
#vector of row numbers
rows = np.arange(local_range[0],local_range[1],dtype=np.int32)
####################################################################
####################################################################
#from the function spaces and ownership ranges, generate global degrees of freedom
dof_coordinates1=V1.tabulate_dof_coordinates()
dof_coordinates2=V2.tabulate_dof_coordinates()
N_dof_1 = len(dof_coordinates1)   #Warning, this might be hardcoed for 1D subdomains
N_dof_2 = len(dof_coordinates2)
global_dof=CF.cartesian_product_coords(dof_coordinates1,dof_coordinates2)
x = global_dof[:,0]
y = global_dof[:,1]
#why do we add local range?
global_boundary_dofs = CF.fetch_boundary_dofs(V1,V2,dof_coordinates1,dof_coordinates2)+ local_range[0]
####################################################################
####################################################################
#generate any coefficients that depend on the degrees of freedom
alpha = 1.0
kappa = np.exp(alpha*x*y)
###################################################################
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
M.setPreallocationNNZ(M_NNZ)
A.setPreallocationNNZ(M_NNZ)
##################################################################
##################################################################
#Loading routine for mass and stiffness matrix
#set the global mass matrix using CSR
M.setValuesCSR(temp2.indptr,temp2.indices,temp2.data)
M.assemble()
#print(M.getOwnershipRange())
#print(M.getValuesCSR())
#Loading A matrix
kappa_func = Function(V1)
#this stores functions to be intergrated
#this is terribly inneficient, maybe look into scipy sparse
#or something like this

#loop through dof_2 and get a N_dof_1xN_dof_1 sparse matrix, need way to store
#is appending the best way?
A1 = []
A2 = []
#need to loop over nodes in N-dof-2
for a in range(N_dof_2):
    #need value at a specific dof_coordinate in second domain
    kappa_func.vector()[:] = np.array(kappa[a::N_dof_2])
    #create expressions and assemble linear forms
    K11 = kappa_func*u1.dx(0)*v1.dx(0)*dx
    K12 = kappa_func*u1*v1*dx
    #then save all matrices to list of matrices
    #since these are sparse maybe take PETSc output and pipe 
    #to scipy sparse matrices
    #maybe not so easy to program on second loop though
    
    K1=PETScMatrix()
    K2 = PETScMatrix()
    K11 = assemble(K11,tensor=K1)

    K12 = assemble(K12,tensor=K2)
    A1.append(K1)
    A2.append(K2)
print(A2[1].mat().getValuesCSR())
'''
#now looping over each submatrix
#obviously looping through each entry is incredibly inefficient but is left this way for simplicity
#since this is just a small demo case. In practice, take advantage of sparsity
fy = Function(V2)
for i in range(N_dof_1):
    for j in range(N_dof_1):
        fy.vector()[:] = np.array(A1[i,j,:])
        K21 = u2*v2*fy*dx
        K21 = assemble(K21)
        K21 = np.array(K21.array())
        
        
        fy.vector()[:] = np.array(A2[i,j,:])
        K22 = u2.dx(0)*v2.dx(0)*fy*dx
        K22 = assemble(K22)
        K22 = np.array(K22.array())

        temp = sp.csr_matrix(K21+K22)

        A.setValuesLocalCSR(np.append(np.zeros(i*N_dof_1),temp.indptr).astype(np.int32),temp.indices+j*N_dof_2,temp.data)
#need to wipe out row and set as the row of the identity matrix
#print(global_boundary_dofs)
A.assemble()
#this may be slow idk since it is after assembly
A.zeroRows(global_boundary_dofs,diag=1)
##################################################################
##################################################################
#assmble RHS
#now evaluate RHS at all d.o.f and set that as the vector
F_dof = PETSc.Vec()
F_dof.create(comm=comm)
F_dof.setSizes((local_rows,global_rows),bsize=1)
F_dof.setFromOptions()

#calculate pointwise values of RHS and put them in F_dof
temp = -2*alpha**2*(x**2+y**2)*np.exp(2*alpha*x*y)
F_dof.setValues(rows,temp)
#now matrix vector multiply with M
B = F_dof.duplicate()
B.setFromOptions()
#Mass matrix
M.mult(F_dof,B)
#lastly, set exact solution at dirichlet boundary
u_true=np.exp(alpha*x*y)
u_d = u_true[global_boundary_dofs-local_range[0]]
B.setValues(global_boundary_dofs,u_d)
#now we should be able to solve for final solution
u_cart = B.duplicate()
pc2 = PETSc.PC().create()
pc2.setType('lu')
pc2.setOperators(A)
ksp2 = PETSc.KSP().create() # creating a KSP object named ksp
ksp2.setOperators(A)
ksp2.setPC(pc2)
B.assemble()
ksp2.solve(B, u_cart)
print('Cartesian solution')
print(u_cart.getArray()[:])

#for verification just print exact
print('Exact')
print(u_true[:])

# Plot solution and mesh
#plot(u)
#plot(mesh)
#plt.savefig('poisson/final_sol_p00'+str(rank)+'.png')
# Save solution to file in VTK format
#vtkfile = File('poisson/solution.pvd')
#vtkfile << u

# Compute error in L2 norm
#error_L2 = errornorm(u_D1, u, 'L2')

# Compute maximum error at vertices
#vertex_values_u_D = u_D1.compute_vertex_values(mesh1)
#vertex_values_u = u.compute_vertex_values(mesh1)
#import numpy as np
#error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
#print('error_L2  =', error_L2)
#print('error_max =', error_max)

# Hold plot
#plt.savefig('poisson/final_sol.png')
'''
