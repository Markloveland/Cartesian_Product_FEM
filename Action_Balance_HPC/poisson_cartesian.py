"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.
  -Laplace(u) = f    in the unit interval
            u = u_D  on the boundary
  u_D = 1 + x^2
    f = -2

see if i can do tensor product and see what happens

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
nx = 1000
ny = 20
#set initial time
t = 0
#set time step


####################################################################
#Subdomain 1
#the first subdomain will be split amongst processors
# this is set up ideally for subdomain 1 > subdomain 2
mesh1 = UnitIntervalMesh(comm,nx)
V1 = FunctionSpace(mesh1, 'P', 1)
# Define boundary condition 1
u_D1 = Expression('1 + x[0]*x[0]', degree=2)
def boundary(x, on_boundary):
    return on_boundary
bc1 = DirichletBC(V1, u_D1, boundary)

# offload this to CF routine maybe
# Define variational problem to generate submatrices for domain 1
u1 = TrialFunction(V1)
v1 = TestFunction(V1)
f1 = Constant(-2.0)
a1 = u1.dx(0)*v1.dx(0)*dx
L1 = f1*v1*dx
K1_pet = PETScMatrix()

#assemble subdomain matrix
assemble(a1,tensor=K1_pet)
K1_sizes = K1_pet.mat().getLocalSize()
K1_global_size =K1_pet.mat().getSize() 
#also need mass matrices in each subdomain
m1 = u1*v1*dx
M1_pet = PETScMatrix()
assemble(m1,tensor=M1_pet)
print(M1_pet.mat().getValuesCSR())
print(M1_pet.mat().getOwnershipRange())
#assmble RHS
f1_pet = PETScVector()
assemble(L1,tensor=f1_pet)
#see if i can apply boundary conditions, I know globally i need to do something else
bc1.apply(K1_pet)
bc1.apply(f1_pet)
u=Function(V1)



#####################################################################
#Subdomain 2
#now we want entire second subdomain on EACH processor, so this will always be the smaller mesh
#MPI.COMM_SELF to not partition mesh
mesh2 = UnitIntervalMesh(MPI.COMM_SELF,ny)
V2 = FunctionSpace(mesh2, 'P', 1)
# Define variational problem
u2 = TrialFunction(V2)
v2 = TestFunction(V2)
f2 = Constant(1.0)
a2 = u2.dx(0)*v2.dx(0)*dx
L2 = f2*v2*dx
    
#assemble, want same global matrix on every  processor
K2_pet = PETScMatrix(MPI.COMM_SELF)
assemble(a2,tensor=K2_pet)
K2_sizes = K2_pet.mat().getLocalSize()


#also need mass matrices in each subdomain
m2 = u2*v2*dx
M2_pet = PETScMatrix(MPI.COMM_SELF)
assemble(m2,tensor=M2_pet)



###############################################################
#Assemble global matrices (product of each subdomain)
local_rows = int(K1_sizes[0]*K2_sizes[0])
global_rows = int(K1_global_size[0]*K2_sizes[0])
local_cols = int(K1_sizes[1]*K2_sizes[1])
global_cols = int(K1_global_size[1]*K2_sizes[1])
#now lets create a global matrix which will be stored on each process
#global matrix is K11 x K22 + K12 x K21

#first I need to create an mpi matrix of the appropriate size and start storing values
A = PETSc.Mat()
A.create(comm=comm)
A.setSizes(([local_rows,global_rows],[local_cols,global_cols]),bsize=1)
A.setFromOptions()
A.setType('aij')
A.setUp()
#also need global mass matrix
#same exact structure as A
M = A.duplicate()

#need to preallocate in here
print('Tensor Product matrix:')
print(A.getSize())
local_range = A.getOwnershipRange()
print(local_range)
#vector of row numbers
rows = np.arange(local_range[0],local_range[1],dtype=np.int32)
#i will also need to calculate the global degrees of freedom
dof_coordinates1=V1.tabulate_dof_coordinates()
dof_coordinates2=V2.tabulate_dof_coordinates()
global_dof=CF.cartesian_product_coords(dof_coordinates1,dof_coordinates2)
#print(global_dof)
x = global_dof[:,0]
y = global_dof[:,1]
global_boundary_dofs = CF.fetch_boundary_dofs(V1,V2,dof_coordinates1,dof_coordinates2)+ local_range[0]


#########################################################
#Loading routine
#preallocate with nnz first
#to get nnz of cartesian matrix, need nnz of each submatrix
#K matrices are u'v'
K1_I,K1_J,K1_A = K1_pet.mat().getValuesCSR()
K1_NNZ = K1_I[1:]-K1_I[:-1]
K2_I,K2_J,K2_A = K2_pet.mat().getValuesCSR()
K2_NNZ = K2_I[1:]-K2_I[:-1]
#same for mass matrices uv
M1_I,M1_J,M1_A = M1_pet.mat().getValuesCSR()
M1_NNZ = M1_I[1:]-M1_I[:-1]
M2_I,M2_J,M2_A = M2_pet.mat().getValuesCSR()
M2_NNZ = M2_I[1:]-M2_I[:-1]
#now use scipy for sparse kron and dump into petsc matrices
K1 = sp.csr_matrix((K1_A,K1_J,K1_I),shape=(K1_sizes[0],K1_global_size[1]))
K2 = sp.csr_matrix((K2_A,K2_J,K2_I),shape=K2_sizes)
M1 = sp.csr_matrix((M1_A,M1_J,M1_I),shape=(K1_sizes[0],K1_global_size[1]))
M2 = sp.csr_matrix((M2_A,M2_J,M2_I),shape=K2_sizes)

temp = sp.kron(K1,M2,format="csr")+sp.kron(M1,K2,format="csr")
A_NNZ = temp.indptr[1:]-temp.indptr[:-1]


temp2 = sp.kron(M1,M2,format="csr")
M_NNZ = temp2.indptr[1:]-temp2.indptr[:-1]

#now need to mass matrixes for stiffness and RHS
M.setPreallocationNNZ(M_NNZ)
#set the global matrix using CSR
M.setValuesCSR(temp2.indptr,temp2.indices,temp2.data)
M.assemble()

#Loading A matrix
A.setPreallocationNNZ(A_NNZ)
A.setValuesCSR(temp.indptr,temp.indices,temp.data)

#need to wipe out row and set as the row of the identity matrix
#print(global_boundary_dofs)
A.assemble()
#this may be slow idk since it is after assembly
A.zeroRows(global_boundary_dofs,diag=1)
#checkout global matrix
#print(A.getInfo())


#now evaluate RHS at all d.o.f and set that as the vector
F_dof = PETSc.Vec()
F_dof.create(comm=comm)
F_dof.setSizes((local_rows,global_rows),bsize=1)
F_dof.setFromOptions()

#calculate pointwise values of RHS and put them in F_dof
temp = -2*np.ones(local_rows)
F_dof.setValues(rows,temp)
#now matrix vector multiply with M
B = F_dof.duplicate()
B.setFromOptions()
#Mass matrix
M.mult(F_dof,B)
#lastly, set exact solution at dirichlet boundary
u_d = 1 + x[global_boundary_dofs-local_range[0]]**2
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
print(u_cart.getArray()[::ny+1])

#for verification just solve problem in subdomain 1
#this is fenics solve
#need equivalent solve step in PETSc
solve(K1_pet,u.vector(),f1_pet)
print('Fenics')
print(u.vector()[:])

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

