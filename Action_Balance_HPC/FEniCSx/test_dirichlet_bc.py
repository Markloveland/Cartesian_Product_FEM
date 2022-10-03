from mpi4py import MPI
from dolfinx import mesh
from dolfinx.fem import FunctionSpace
from dolfinx import fem,cpp
import numpy
import ufl
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import numpy as np


#get MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

domain = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8, mesh.CellType.triangle)
V = FunctionSpace(domain, ("CG", 1))
uD = fem.Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(uD, boundary_dofs)

#try to do it manually in petsc and get same result
#print(boundary_dofs)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain, ScalarType(-6))
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

problem_no_bc = fem.petsc.LinearProblem(a,L)
problem_no_bc.solve()
problem = fem.petsc.LinearProblem(a, L,bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()


#compute error
V2 = fem.FunctionSpace(domain, ("CG", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
error_max = numpy.max(numpy.abs(uD.x.array-uh.x.array))
# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")





#tryto reproduce manually
bilinear_form = fem.form(a)
linear_form = fem.form(L)
A = fem.petsc.assemble_matrix(bilinear_form)
A.assemble()
#something weird happening here????
bf = fem.petsc.create_vector(linear_form)
fem.petsc.assemble_vector(bf, linear_form)
#print(type(A))


rows = A.getOwnershipRange()
dum = np.zeros(rows[1]-rows[0])
dofs = V.tabulate_dof_coordinates()

local_range1 = V.dofmap.index_map.local_range

dofs1 = np.arange(*local_range1,dtype=np.int32)
local_size1 = V.dofmap.index_map.size_local
x_local1 = np.array([dofs[:local_size1,0]]).T
y_local1 = np.array([dofs[:local_size1,1]]).T
global_size1 = V.dofmap.index_map.size_global


A_bnd = PETSc.Mat()
A_bnd.create(comm=MPI.COMM_WORLD)
A_bnd.setSizes(([local_size1,global_size1],[local_size1,global_size1]),bsize=1)
A_bnd.setFromOptions()
A_bnd.setType('aij')
A_bnd.setUp()


temp = PETSc.Vec()
temp.create(comm=comm)
temp.setSizes((local_size1,global_size1),bsize=1)
temp.setFromOptions()
b = temp.duplicate()
u_sol = temp.duplicate()
i=0

#original boundary from fenics
#print(boundary_dofs)

print('ghosted nodes',rank,boundary_dofs[boundary_dofs>=local_size1])
#need to eliminate ghost nodes
boundary_dofs = boundary_dofs[boundary_dofs<local_size1]
print('boundary dofs, rank ',rank,boundary_dofs)


#print(boundary_dofs)
#print(boundary_dofs)

x_dof = dofs[boundary_dofs,0]
y_dof = dofs[boundary_dofs,1]
u_D = 1 +x_dof**2 + 2*y_dof**2
boundary_dofs = boundary_dofs+rows[0]

#now go in and fill -1 for rows and columns of boundary nodes
#collect
all_boundary_dofs = np.concatenate(MPI.COMM_WORLD.allgather(boundary_dofs))
print("collected boundary dofs",all_boundary_dofs)
all_u_D = np.concatenate(MPI.COMM_WORLD.allgather(u_D))
for col in all_boundary_dofs:
    A.getColumnVector(col,temp)
    dum = dum + all_u_D[i]*temp.getArray()
    i=i+1

temp.setValues(range(rows[0],rows[1]),dum)



#something is wrong with RHS
dum = problem_no_bc.b.getValues(dofs1)
b.setValues(dofs1,dum)
b.assemble()

#nned to also multiply by exact solution at node
b = b - temp

#also fix the boundary nodes at the correct values
b.setValues(boundary_dofs,u_D)
#A.zeroRows(boundary_dofs,diag=1)
A.zeroRowsColumns(boundary_dofs,diag=1)
b.assemble()
#A.zeroRowsColumns(boundary_dofs)
#now solve and compare solution



pc2 = PETSc.PC().create()
#this is a direct solve with lu
pc2.setType('lu')
pc2.setOperators(A)
ksp2 = PETSc.KSP().create() # creating a KSP object named ksp
ksp2.setOperators(A)
#ksp2.setType('gmres')
ksp2.setPC(pc2)
ksp2.solve(b, u_sol)

diff1 = problem.b - b
diff2 = A-problem.A


#if rank == 0:
#    print('b',problem.b.getValues(dofs1))
#    print(b.getValues(dofs1))
     #print(u_sol.getValues(dofs1))
print('difference in RHS on rank',rank,diff1.getArray())
#print('difference in Matix on rank',rank,diff2.getValues(dofs1,dofs1))
print('total sum',rank,np.sum(np.abs(diff2.getValues(dofs1,dofs1))))
if rank == 2:
    print('problem',problem.A.getValues(range(rows[0],rows[0]+5),range(rows[0],rows[0]+5)))
    print('my version',A.getValues(range(rows[0],rows[0]+5),range(rows[0],rows[0]+5)))
    print('rows:',rows[0])
#print(uh.vector.getValues(dofs1))
#compute error
V2 = fem.FunctionSpace(domain, ("CG", 2))
uex = fem.Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)

#this includes ghost
#print(uh.x.array[:].shape)
#this does not
uh.vector.setValues(range(rows[0],rows[1]),u_sol.getArray())
uh.vector.ghostUpdate()

L2_error = fem.form(ufl.inner(uh - uex, uh - uex) * ufl.dx)
error_local = fem.assemble_scalar(L2_error)
error_L2 = numpy.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
error_max = numpy.max(numpy.abs(uD.x.array-uh.x.array))
# Only print the error on one process
if domain.comm.rank == 0:
    print(f"Error_L2 : {error_L2:.2e}")
    print(f"Error_max : {error_max:.2e}")


