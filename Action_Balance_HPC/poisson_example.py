"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.
  -Laplace(u) = f    in the unit square
            u = u_D  on the boundary
  u_D = 1 + x^2 + 2y^2
    f = -6
"""

from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# Create mesh and define function space
nx = 100
mesh = UnitSquareMesh(nx, nx)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
a = dot(grad(u), grad(v))*dx
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
K = assemble(a)
#print(dir(K))
#print(type(K))
b = assemble(L)
bc.apply(K)
bc.apply(b)
#print(K.array().shape)
u=Function(V)
#splits it up by rows and solves global system in a distributed way automatically!!!!
#solve(K,u.vector(),b)

# Compute solution
u = Function(V)
solve(a == L, u, bc)

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
