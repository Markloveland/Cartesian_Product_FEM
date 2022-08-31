import dolfinx
import dolfinx
from mpi4py import MPI
from dolfinx import mesh
from dolfinx import fem
from dolfinx import cpp
import numpy as np
import ufl
from petsc4py.PETSc import ScalarType
from petsc4py import PETSc
import CFx.transforms
import CFx.boundary
import CFx.assemble
import CFx.wave
##############
#Domain 1
domain1 = mesh.create_interval(MPI.COMM_WORLD,5,(0,1))
#domain = mesh.create_interval(MPI.COMM_WORLD,25,(0,1),  ghost_mode=dolfinx.cpp.mesh.GhostMode.none)
V1 = fem.FunctionSpace(domain1, ("CG", 1))
dof_coords1 = V1.tabulate_dof_coordinates()
print('domain1 coords',dof_coords1)

local_range1 = V1.dofmap.index_map.local_range
dofs1 = np.arange(*local_range1,dtype=np.int32)

local_size1 = V1.dofmap.index_map.size_local
x_local1 = np.array([dof_coords1[:local_size1,0]]).T
global_size1 = V1.dofmap.index_map.size_global
###############
###############
#Domain 2
domain2 = mesh.create_interval(MPI.COMM_SELF,4,(0,1))
#domain = mesh.create_interval(MPI.COMM_WORLD,25,(0,1),  ghost_mode=dolfinx.cpp.mesh.GhostMode.none)
V2 = fem.FunctionSpace(domain2, ("CG", 1))
dof_coords2 = V2.tabulate_dof_coordinates()
print('domain2 coords',dof_coords2)

local_range2 = V2.dofmap.index_map.local_range
dofs2 = np.arange(*local_range1,dtype=np.int32)

local_size2 = V2.dofmap.index_map.size_local
x_local2 = np.array([dof_coords2[:local_size2,0]]).T
global_size2 = V2.dofmap.index_map.size_global
################
################

#now testing imports of new cartesian functions library
new_coords = CFx.transforms.cartesian_product_coords(x_local1,x_local2)
b_dof = CFx.boundary.fetch_boundary_dofs(domain1,domain2,V1,V2,local_size1,local_size2)

print('new coords',new_coords)
print('new_coords shpe',new_coords.shape)
print('domain boundary',b_dof)



#######
#testing computation of significant wave height
F_dof = PETSc.Vec()
F_dof.create(comm=MPI.COMM_WORLD)
F_dof.setSizes((local_size1*local_size2,global_size1*global_size2),bsize=1)
F_dof.setFromOptions()

#let's set F as constant first
#get ownership range
local_range = F_dof.getOwnershipRange()
#vector of row numbers
rows = np.arange(local_range[0],local_range[1],dtype=np.int32)

F_dof.setValues(rows,2*np.ones(len(rows)))

HS_vec = CFx.wave.calculate_HS(F_dof,V2,local_size1,local_size2,local_range2)

print('HS',HS_vec)
