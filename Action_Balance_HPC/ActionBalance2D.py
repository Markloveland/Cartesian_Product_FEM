"""
Action Balance Equation Solver
This algorithm for loading will be same required of Action Balance Equation
    du/dt + \/.cu = f
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
#soecify domain size
L = 10
# Create cartesian mesh of two 2D and define function spaces
nx = 19
ny = 19
#set initial time
t = 0
#set final time
t_f = 1/200
#set time step
dt = 1/200
#calculate nt
nt = int(np.ceil(t_f/dt))

####################################################################
#Subdomain 1
#the first subdomain will be split amongst processors
# this is set up ideally for subdomain 1 > subdomain 2
mesh1 = IntervalMesh(comm,nx,0.0,L)
V1 = FunctionSpace(mesh1, 'P', 1)
u1 = TrialFunction(V1)
v1 = TestFunction(V1)
####################################################################
####################################################################
#Subdomain 2
#now we want entire second subdomain on EACH processor, so this will always be the smaller mesh
#MPI.COMM_SELF to not partition mesh
mesh2 =  IntervalMesh(MPI.COMM_SELF,ny,0.0,L)
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
#save sizes of subdomain matrices
M1_sizes = M1_pet.mat().getLocalSize()
M1_global_size =M1_pet.mat().getSize() 
M2_sizes = M2_pet.mat().getLocalSize()

#calculate sizes of matrix for global domain
#number of rows/cols on each processor
local_rows = int(M1_sizes[0]*M2_sizes[0])
global_rows = int(M1_global_size[0]*M2_sizes[0])
local_cols = int(M1_sizes[1]*M2_sizes[1])
global_cols = int(M1_global_size[1]*M2_sizes[1])
###################################################################
###################################################################
#Allocate global mass matrix
#need to generate global mass matrix to get global matrix layout and sparsity patterns
#global matrices are product of each subdomain
M=CF.create_cartesian_mass_matrix(local_rows,global_rows,local_cols,global_cols)

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
N_dof_1 = dof_coordinates1.shape[0]   #Warning, this might be hardcoed for 1D subdomains
N_dof_2 = dof_coordinates2.shape[0]   
local_dof=CF.cartesian_product_coords(dof_coordinates1,dof_coordinates2)
x = local_dof[:,0]
y = local_dof[:,1]
#get global equation number of any node on entire global boundary
local_boundary_dofs = CF.fetch_boundary_dofs(V1,V2,dof_coordinates1,dof_coordinates2)

#now only want subset that is the inflow, need to automate later
x_min = 0
y_min = 0
dum1 = local_boundary_dofs[x[local_boundary_dofs]<=(x_min+1e-14)]
dum2 = local_boundary_dofs[y[local_boundary_dofs]<=(y_min+1e-14)]
local_boundary_dofs = np.unique(np.concatenate((dum1,dum2),0))
#local_boundary_dofs=dum2
global_boundary_dofs = local_boundary_dofs + local_range[0]
#print('global_boundary_dofs')
#print(global_boundary_dofs)
#print('locations of the boundary')
#print(x[local_boundary_dofs])
#print(y[local_boundary_dofs])
#check mesh dof are agreeing
#print('Mesh1 dof shape and x,y')
#print(dof_coordinates1.shape)
#print(dof_coordinates1[:,0])
#print(dof_coordinates1[:,1])
#print('Calculated x')
#print(x[::N_dof_2])
#print('Calculated y')
#print(y[::N_dof_2])

####################################################################
####################################################################
#generate any coefficients that depend on the degrees of freedom
c = np.zeros(local_dof.shape)
c[:,:] = 1.0
#exact solution and dirichlet boundary
u_true=np.sin(x-c[:,0]*t) + np.cos(y-c[:,1]*t)
u_2 = np.sin(x-c[:,0]*(t+dt)) + np.cos(y-c[:,1]*(t+dt))
u_d = u_2[local_boundary_dofs]
###################################################################
###################################################################
#Preallocate and load/assemble cartesian mass matrix!
#now need to mass matrixes for stiffness and RHS, also optionally can out put the nnz
M_NNZ = CF.build_cartesian_mass_matrix(M1_pet,M2_pet,M1_sizes,M1_global_size,M2_sizes,M)
A.setPreallocationNNZ(M_NNZ)
##################################################################
##################################################################
#Loading A matrix routine
CF.build_stiffness_varying_action_balance_2D(mesh1,V1,mesh2,V2,c,N_dof_2,dt,A)
A=A+M
#set Dirichlet boundary as global boundary
A.zeroRows(global_boundary_dofs,diag=1)
#just want to test answer
#A.zeroRows(rows,diag=1)
##################################################################
##################################################################
#assmble RHS
#now evaluate RHS at all d.o.f and set that as the F vector
F_dof = PETSc.Vec()
F_dof.create(comm=comm)
F_dof.setSizes((local_rows,global_rows),bsize=1)
F_dof.setFromOptions()

#calculate pointwise values of RHS and put them in F_dof
temp = u_true
F_dof.setValues(rows,temp)

#now matrix vector multiply with M to get actual right hand side
B = F_dof.duplicate()
E = F_dof.duplicate()
L2_E = F_dof.duplicate()
E.setFromOptions()
B.setFromOptions()
L2_E.setFromOptions()
#Mass matrix
M.mult(F_dof,B)
#set Dirichlet boundary conditions
B.setValues(global_boundary_dofs,u_d)
#just want to test answer
#B.setValues(rows,u_2)
###################################################################
###################################################################
#Time step
#u_cart will hold solution
u_cart = B.duplicate()
#create a linear solver
pc2 = PETSc.PC().create()
#this is a direct solve with lu
pc2.setType('lu')
pc2.setOperators(A)
ksp2 = PETSc.KSP().create() # creating a KSP object named ksp
ksp2.setOperators(A)
ksp2.setPC(pc2)
B.assemble()

for i in range(nt):
    t+=dt
    ksp2.solve(B, u_cart)

####################################################################
###################################################################
#Post Processing section

#print whole solution
#print('Cartesian solution')
#print(u_cart.getArray()[:])
#print('Exact')
#print(u_true[:])

u_true= np.sin(x-c[:,0]*t) + np.cos(y-c[:,1]*t)
u_exact = PETSc.Vec()
u_exact.create(comm=comm)
u_exact.setSizes((local_rows,global_rows),bsize=1)
u_exact.setFromOptions()
u_exact.setValues(rows,u_true)

PETSc.Sys.Print("Final t",t)
#need function to evaluate L2 error
e1 = u_cart-u_exact
PETSc.Vec.pointwiseMult(E,e1,e1)
M.mult(E,L2_E)
#L2
PETSc.Sys.Print("L2 error",np.sqrt(L2_E.sum()))
#Linf
PETSc.Sys.Print("L inf error",e1.norm(PETSc.NormType.NORM_INFINITY)) 
#min/max
PETSc.Sys.Print("min in error",e1.min())
PETSc.Sys.Print("max error",e1.max())
#h
PETSc.Sys.Print("h",1/nx)
#dof
PETSc.Sys.Print("dof",(nx+1)*(ny+1))

##################################################################
#################################################################
#If I can find integral parameter like Hs then this section
# could be helpful for visualization


# Save solution to file in VTK format
#u = Function(V1)
#u.vector()[:] = np.array(u_cart.getArray()[4::N_dof_2])
#vtkfile = File('ActionBalance/solution.pvd')
#vtkfile << u


#Only for serial!
#############
#fig,  ax2 = plt.subplots(nrows=1)

#ax2.tricontour(local_dof[:,0], local_dof[:,1], u_cart.getArray()[:], levels=14, linewidths=0.5, colors='k')
#cntr2 = ax2.tricontourf(local_dof[:,0], local_dof[:,1], u_cart.getArray()[:], levels=14)#, cmap="RdBu_r")

#fig.colorbar(cntr2, ax=ax2)
#ax2.plot(local_dof[:,0], local_dof[:,1], 'ko', ms=3)
#ax2.set(xlim=(0, 1), ylim=(0, 1))
#ax2.set_title('Cartesian Product Solution')
#plt.subplots_adjust(hspace=0.5)
#plt.savefig('2DPropagation.png')
#print(x[local_boundary_dofs])
#print(y[local_boundary_dofs])
#print(max(abs(u_exact.getArray()[:]-u_cart.getArray()[:])))
#print(u_exact.getArray()[local_boundary_dofs] - u_cart.getArray()[local_boundary_dofs])
###########


#lets print out array to see what is happening
#print(local_boundary_dofs)
#print(global_boundary_dofs)
#A=A-M
#print(A.getValues(range(16),range(16)))
#print(M.getValues(range(16),range(16)))
# Plot solution and mesh on each individual process
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
#vertex_values_u = #u.compute_vertex_values(mesh1)
#import numpy as np
#error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
#print('error_L2  =', error_L2)
#print('error_max =', error_max)

# Hold plot
#plt.savefig('poisson/final_sol.png')

