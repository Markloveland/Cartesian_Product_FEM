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
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
import cartesianfunctions as CF
# Create cartesian mesh of two intervals and define function spaces
nx = 1000
ny = 1000


#the first subdomain will be split amongst processors
mesh1 = UnitIntervalMesh(comm,nx)
V1 = FunctionSpace(mesh1, 'P', 1)
# Define boundary condition 1
u_D1 = Expression('1 + x[0]*x[0]', degree=2)
def boundary(x, on_boundary):
    return on_boundary
bc1 = DirichletBC(V1, u_D1, boundary)
# Define variational problem to generate submatrices for domain 1
u1 = TrialFunction(V1)
v1 = TestFunction(V1)
f1 = Constant(-2.0)
a1 = u1.dx(0)*v1.dx(0)*dx
L1 = f1*v1*dx
K1_pet = PETScMatrix()
assemble(a1,tensor=K1_pet)
f1_pet = PETScVector()
assemble(L1,tensor=f1_pet)
#see if i can apply boundary conditions, I know globally i need to do something else
bc1.apply(K1_pet)
bc1.apply(f1_pet)
u=Function(V1)
#lets print out each submatrix
#print('Subdomain 1 stiffness matrix')
#print(K1_pet.mat().getValuesCSR())

#some methods to inspect matrix
#print(dir(K_pet.mat()))
#print(K_pet.mat().getInfo())
#print(K_pet.mat().getValuesCSR())
#print('Local then global sizes of K1')
#print(K1_pet.mat().getOwnershipRange())
K1_sizes = K1_pet.mat().getLocalSize()
#print(K1_sizes)
K1_global_size =K1_pet.mat().getSize() 
#print(K1_global_size)

#lets see if we can get a consolidated one and have one on each proccessor


#now we want entire second subdomain on EACH processor, so this will always be the smaller mesh
#MPI.COMM_SELF to not partition mesh??
mesh2 = UnitIntervalMesh(MPI.COMM_SELF,ny)
    
V2 = FunctionSpace(mesh2, 'P', 1)
# Define variational problem
u2 = TrialFunction(V2)
v2 = TestFunction(V2)
f2 = Constant(1.0)
a2 = u2.dx(0)*v2.dx(0)*dx #+ u.dx(1)*v.dx(1)*dx
L2 = f2*v2*dx
    
#assemble, want same global matrix on every  processor
K2_pet = PETScMatrix(MPI.COMM_SELF)
assemble(a2,tensor=K2_pet)
#print("K2 Petscmatrix")
K2_sizes = K2_pet.mat().getLocalSize()
#print('Local Size of K2')
#print(K2_sizes)
#print(K2_pet.mat().getValues(range(ny+1),range(ny+1))  )

#now lets create a global matrix which will be stored on each process
#global matrix is K11 x K22 + K12 x K21

#first I need to create an mpi matrix of the appropriate size and start storing values
A = PETSc.Mat()
A.create(comm=comm)
local_rows = int(K1_sizes[0]*K2_sizes[0])
global_rows = int(K1_global_size[0]*K2_sizes[0])
local_cols = int(K1_sizes[1]*K2_sizes[1])
global_cols = int(K1_global_size[1]*K2_sizes[1])
A.setSizes(([local_rows,global_rows],[local_cols,global_cols]),bsize=1)
A.setFromOptions()
A.setType('aij')
A.setUp()
#need to preallocate in here
print('Tensor Product matrix:')

local_range = A.getOwnershipRange()
#print(local_rows)
#print(A.getLocalSize())
print(A.getSize())

#This should be the right size
#i will also need to calculate the global degrees of freedom
dof_coordinates1=V1.tabulate_dof_coordinates()
dof_coordinates2=V2.tabulate_dof_coordinates()
#print(dof_coordinates1)
#print(dof_coordinates2)
global_dof=CF.cartesian_product_coords(dof_coordinates1,dof_coordinates2)
#print(global_dof)
x = global_dof[:,0]
y = global_dof[:,1]
global_boundary_dofs = CF.fetch_boundary_dofs(V1,V2,dof_coordinates1,dof_coordinates2)+ local_range[0]
#print(global_boundary_dofs)
#need to assign values
#preallocate with nnz first
#######
#to get nnz of cartesian matrix, need nnz of each submatrix

I1,J1,A1 = K1_pet.mat().getValuesCSR()
NNZ1 = I1[1:]-I1[:-1]
I2,J2,A2 = K2_pet.mat().getValuesCSR()
#print('CSR matrix 1:')
#print(I1)
#print(J1)
#print(A1)
#print('CSR matrix 2:')
#print(I2)
#print(J2)
#print(A2)
NNZ2 = I2[1:]-I2[:-1]
NNZ = np.kron(NNZ1,NNZ2)
#print(NNZ1)
#print(NNZ2)
#print(NNZ)
A.setPreallocationNNZ(NNZ)

#now use scipy to perform kron on csr matrices
K1 = sp.csr_matrix((A1,J1,I1),shape=(K1_sizes[0],K1_global_size[1]))
K2 = sp.csr_matrix((A2,J2,I2),shape=K2_sizes)
temp = sp.kron(K1,K2,format="csr")
#print('Indices, values, rows of global mat')
#print(temp.indices)
#print(temp.data)
#print(temp.indptr)
#set the global matrix using CSR
A.setValuesCSR(temp.indptr,temp.indices,temp.data)

rows = np.arange(local_range[0],local_range[1],dtype=np.int32)
#for a in rows:
#    A.setValues(a,a, 0.0)

#this would be the identity matrix for the boundary dofs
#need to wipe out row and set as the row of the identity matrix
#print(global_boundary_dofs)
#print(type(global_boundary_dofs[0]))
A.assemble()
#this may be slow idk since it is after assembly
A.zeroRows(global_boundary_dofs,diag=1)
#checkout global matrix
#print(A.getInfo())
#view global matrix
#print(rows)
#print(type(rows))
#print(type(rows[0]))
#print(A.getValues(rows,rows))


#now need to construct RHS
#first need global mass matrix
m1 = u1*v1*dx
M1_pet = PETScMatrix()
assemble(m1,tensor=M1_pet)
m2 = u2*v2*dx
M2_pet = PETScMatrix(MPI.COMM_SELF)
assemble(m2,tensor=M2_pet)
I1,J1,A1 = M1_pet.mat().getValuesCSR()
NNZ1 = I1[1:]-I1[:-1]
I2,J2,A2 = K2_pet.mat().getValuesCSR()
NNZ2 = I2[1:]-I2[:-1]
NNZ = np.kron(NNZ1,NNZ2)


M = PETSc.Mat()
M.create(comm=comm)
#should be same as stiffness matrix
M.setSizes(([local_rows,global_rows],[local_cols,global_cols]),bsize=1)
M.setFromOptions()
M.setType('aij')
M.setUp()
M.setPreallocationNNZ(NNZ)
K1 = sp.csr_matrix((A1,J1,I1),shape=(K1_sizes[0],K1_global_size[1]))
K2 = sp.csr_matrix((A2,J2,I2),shape=K2_sizes)
temp = sp.kron(K1,K2,format="csr")
#set the global matrix using CSR
M.setValuesCSR(temp.indptr,temp.indices,temp.data)
M.assemble()


#now evaluate RHS at all d.o.f and set that as the vector
F_dof = PETSc.Vec()
F_dof.create(comm=comm)
F_dof.setSizes((local_rows,global_rows),bsize=1)
F_dof.setFromOptions()
#these should be same
#print(F_dof.getOwnershipRange())
#print(local_range)

#calculate pointwise values of RHS and put them in F_dof
temp = -2*np.ones(local_rows)
F_dof.setValues(rows,temp)
#now matrix vector multiply with M
B = F_dof.duplicate()
B.setFromOptions()
#Mass matrix
M.mult(F_dof,B)
#print('Mass matrix')
#print(M.getValues(rows,rows))
#print('F_dof')
#print(F_dof.getArray())
#print('M*F = ')
#print(B.getArray())
#lastly, set exact solution at dirichlet boundary
u_d = 1 + x[global_boundary_dofs-local_range[0]]**2
B.setValues(global_boundary_dofs,u_d)
#print('B after boundaries')
#print(B.getArray())
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
print(u_cart.getArray())


#this is fenics solve
#need equivalent solve step in PETSc
#K1_pet.mat()=A
solve(K1_pet,u.vector(),f1_pet)
print('Fenics')
print(u.vector()[:])

#PETSc solve
RHS_pet = f1_pet.vec()
u = RHS_pet.duplicate()
pc1 = PETSc.PC().create()
pc1.setType('lu')
pc1.setOperators(K1_pet.mat())
ksp = PETSc.KSP().create() # creating a KSP object named ksp
ksp.setOperators(K1_pet.mat())
ksp.setPC(pc1)
ksp.solve(RHS_pet, u)
print('Petsc')
print(u.getArray())



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

