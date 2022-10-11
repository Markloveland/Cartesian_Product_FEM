from mpi4py import MPI
from dolfinx import mesh
from dolfinx.fem import FunctionSpace
from dolfinx import fem
from dolfinx import io
from dolfinx import cpp
import numpy as np
import ufl
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc

#specify bouns in geographic mesh
x_min = 0.0
x_max = 4000
y_min = 0.0
y_max = 20000
# Create cartesian mesh of two 2D and define function spaces
nx = 100
ny = 100

#domain = mesh.create_unit_square(MPI.COMM_WORLD, 12, 12, mesh.CellType.triangle)#, ghost_mode=cpp.mesh.GhostMode.none)
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([x_min, y_min]), np.array([x_max, y_max])], [nx, ny], mesh.CellType.triangle)

V = FunctionSpace(domain, ("CG", 1))


#alternatively, try to define pointwise
#this gives ghost and owned dof coords
dof_coords = V.tabulate_dof_coordinates()
#suggested in forum, gives index of dofs I want
local_range = V.dofmap.index_map.local_range
#vector of indexes that we want
dofs = np.arange(*local_range,dtype=np.int32)
#gets number of dofs owned
local_size = V.dofmap.index_map.size_local
#hopefully the dof coordinates owned by the process
local_dof_coords = dof_coords[0:local_size,:]

#try to set proper values
#uD.vector.setValues(dofs,  1 + local_dof_coords[:,0]**2 + 2*local_dof_coords[:,1]**2)
#also need to propagate ghost values
#uD.vector.ghostUpdate()

#the above should be same as these two commands combined
# This call takes the values from the ghost regions and accumulates (adds) them to the owning process.
#uD.vector.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
# Important point: The ghosts are still inconsistent!
# This call takes the values from the owning processes and updates the ghosts.
#uD.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


# Create facet to cell connectivity required to determine boundary facets


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)


#f = fem.Constant(domain, ScalarType(-6))
#a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
f = fem.Function(V)
f.x.array[:] = 20 - dof_coords[:,0]/200

#f.interpolate(lambda x:-x[0]/200)
a = u*v*ufl.dx
L = f.dx(0) * v * ufl.dx

problem = fem.petsc.LinearProblem(a, L)#, bcs=[bc])#, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
print(uh.x.array[:])
'''
V2 = fem.FunctionSpace(domain, ("CG", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))

error_max = np.max(np.abs(uD.x.array-uh.x.array))
# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")

from dolfinx import io
with io.VTKFile(domain.comm, "output.pvd", "w") as vtk:
    vtk.write([uh._cpp_object])
with io.XDMFFile(domain.comm, "output.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)
'''
