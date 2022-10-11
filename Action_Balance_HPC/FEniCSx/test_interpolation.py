from mpi4py import MPI
from dolfinx import mesh
from dolfinx import fem
from dolfinx import io
from dolfinx import cpp
import numpy as np
import ufl
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc

#get MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

#specify bouns in geographic mesh
x_min = 0.0
x_max = 4000
y_min = 0.0
y_max = 20000
# Create cartesian mesh of two 2D and define function spaces
nx = 100
ny = 100

domain = mesh.create_rectangle(comm, [np.array([x_min, y_min]), np.array([x_max, y_max])], [nx, ny], mesh.CellType.triangle)
V1 = fem.FunctionSpace(domain, ("CG", 1))

f = fem.Function(V1)
#from the function spaces and ownership ranges, generate global degrees of freedom
#this gives ghost and owned dof coords
dof_coords1 = V1.tabulate_dof_coordinates()
#suggested in forum, gives index of dofs I want
local_range1 = V1.dofmap.index_map.local_range
#vector of indexes that we want
dofs1 = np.arange(*local_range1,dtype=np.int32)
#gets number of dofs owned
N_dof_1 = V1.dofmap.index_map.size_local
#hopefully the dof coordinates owned by the process
local_dof_coords1 = dof_coords1[0:N_dof_1,:domain.topology.dim]
#for now lets set depth as x coordinate itself
#eventually this maybe read in from txt file or shallow water model
#this includes ghost nodes!
#f.interpolate(lambda x: 20 - x[0]/200)






u = ufl.TrialFunction(V1)
v = ufl.TestFunction(V1)


f.x.array[:] = -dof_coords1[:,0]/200

'''
#f.interpolate(lambda x:-x[0]/200)
a = u*v*ufl.dx
L = f.dx(0) * v * ufl.dx

problem = fem.petsc.LinearProblem(a, L)#, bcs=[bc])#, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
print(uh.x.array[:])


'''
a = u*v* ufl.dx
f.x.array[:] = -dof_coords1[:,0]/200
L = f.dx(0)*v*ufl.dx
problem = fem.petsc.LinearProblem(a, L)#, petsc_options={"ksp_type": "gmres"})
ux = problem.solve()

print(ux.x.array[:])


if rank==0:
    print(rank)
    #print(ux.vector.getArray())

#L2 = f.dx(1)*v*ufl.dx
#problem = fem.petsc.LinearProblem(a,L2,petsc_options={"ksp_type":"gmres"})
#uy = problem.solve

