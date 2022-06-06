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
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
# Create mesh and define function space
nx = 10
mesh = UnitIntervalMesh(comm,nx)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('1 + x[0]*x[0]', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-2.0)
a = u.dx(0)*v.dx(0)*dx #+ u.dx(1)*v.dx(1)*dx
L = f*v*dx

#see what a mass matrix looks like in parallel
'''
M = u*v*dx
K11_pet = PETScMatrix()
assemble(M,tensor = K11_pet)
print('Hi')
print(rank)
K11_pet = K11_pet.mat()
#print(K11_pet.getInfo())

print(K11_pet.getType())
print(K11_pet.getSize())
print(K11_pet.getLocalSize())
print(K11_pet.getValuesCSR())
#print(dir(K11_pet))
#print(K11_pet.getValues(range((nx+1)*(nx+1)) ,range((nx+1)*(nx+1))) )
'''


#try linear algebra  version
#K = assemble(a)
#print(dir(K))
#print(type(K))
#b = assemble(L)
#bc.apply(K)
#bc.apply(b)
#print(K.array().shape)
#u=Function(V)
#splits it up by rows and solves global system in a distributed way automatically!!!!
#solve(K,u.vector(),b)

# Compute solution
#u = Function(V)
#solve(a == L, u, bc)


#try petsc version
#distributed matrix
K_pet = PETScMatrix()
assemble(a,tensor=K_pet)
f_pet = PETScVector()
assemble(L,tensor=f_pet)
#see if i can apply boundary conditions
bc.apply(K_pet)
#print(dir(K_pet))  
bc.apply(f_pet)
u=Function(V)
#print(dir(K_pet.mat()))
#print(K_pet.mat().getInfo())
#print(K_pet.mat().getValuesCSR())
#print(K_pet.mat().getLocalSize())
#print(K_pet.mat().getSize())
#lets see if we can get a consolidated one and have one on each proccessor
#K2_pet = PETScMatrix()
#K2_pet = K2_pet.mat()

#if rank<=nprocs:
print('Hi')
    
nx2 = 8
#MPI.COMM_SELF to not partition mesh??
mesh2 = UnitIntervalMesh(MPI.COMM_SELF,nx2)
    
V2 = FunctionSpace(mesh2, 'P', 1)
# Define variational problem
u2 = TrialFunction(V2)
v2 = TestFunction(V2)
f2 = Constant(-2.0)
a2 = u2.dx(0)*v2.dx(0)*dx #+ u.dx(1)*v.dx(1)*dx
L2 = f2*v2*dx
    
#ideally want to do something like this
#but it does something strange
#assemble
K2_pet = PETScMatrix(MPI.COMM_SELF)
assemble(a2,tensor=K2_pet)
print("rank ",str(rank))
print("K2 Petscmatrix")
print(K2_pet.mat().getValuesCSR())
print(K2_pet.mat().getInfo())
print(K2_pet.mat().getLocalSize())
print(K2_pet.mat().getSize())

print(K2_pet.mat().getValues(range(nx2),range(nx2))  )
#numpy version
#K2 = assemble(a2).array()   
#print('K2 mat values:')
#print(K2)
#print('K2 size')
#print(K2.shape)

#now lets create a global matrix which will be stored on each process
#by broadcasting



solve(K_pet,u.vector(),f_pet)

# Plot solution and mesh
#plot(u)
#plot(mesh)
#plt.savefig('poisson/final_sol_p00'+str(rank)+'.png')
# Save solution to file in VTK format
#vtkfile = File('poisson/solution.pvd')
#vtkfile << u

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
import numpy as np
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
print('error_L2  =', error_L2)
print('error_max =', error_max)

# Hold plot
#plt.savefig('poisson/final_sol.png')

