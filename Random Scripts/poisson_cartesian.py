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
import scipy
import petsc4py
petsc4py.init()
from petsc4py import PETSc
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
import cartesianfunctions as CF
# Create cartesian mesh of two intervals and define function spaces
nx = 10
ny = 8


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
print('Subdomain 1 stiffness matrix')
print(K1_pet.mat().getValuesCSR())

#some methods to inspect matrix
#print(dir(K_pet.mat()))
#print(K_pet.mat().getInfo())
#print(K_pet.mat().getValuesCSR())
print('Local then global sizes of K1')
print(K1_pet.mat().getOwnershipRange())
K1_sizes = K1_pet.mat().getLocalSize()
print(K1_sizes)
K1_global_size =K1_pet.mat().getSize() 
print(K1_global_size)

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
A.setSizes(([local_rows,global_rows],[local_rows,global_rows]),bsize=1)
A.setFromOptions()
A.setType('aij')
A.setUp()
#need to preallocate in here
A.assemble()
print('Tensor Product matrix:')
local_rows = A.getOwnershipRange()
print(local_rows)
print(A.getLocalSize())
print(A.getSize())


#This should be the right size
#i will also need to calculate the global degrees of freedom
dof_coordinates1=V1.tabulate_dof_coordinates()
dof_coordinates2=V2.tabulate_dof_coordinates()
print(dof_coordinates1)
print(dof_coordinates2)
global_dof=CF.cartesian_product_coords(dof_coordinates1,dof_coordinates2)
print(global_dof)
x = global_dof[:,0]
y = global_dof[:,1]
global_boundary_dofs = CF.fetch_boundary_dofs(V1,V2,dof_coordinates1,dof_coordinates2)+ local_rows[0]
print(global_boundary_dofs)
#need to assign values
#preallocate with nnz first
#######



#######
#try a vector first

sizes = np.zeros(nprocs)
sizes[0] = 4
sizes[1] = 7
sizes[2] = 8

X = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
X.setSizes((sizes[rank],PETSc.DECIDE),bsize=1)
X.setFromOptions()
ilow,ihigh = X.getOwnershipRange()

PETSc.Sys.syncPrint("rank: ",rank,"low/high: ",ilow,ihigh)
PETSc.Sys.syncFlush()


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

