"""
Action Balance Equation Solver
This algorithm for loading will be same required of Action Balance Equation
    du/dt + \/.cu = f
"""

import numpy as np
#import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy import sparse as sp
from petsc4py import PETSc
from dolfinx import fem,mesh,io
import ufl
import time
import CFx.assemble
import CFx.transforms
import CFx.boundary
time_start = time.time()



#get MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

#soecify domain size
L = 10
# Create cartesian mesh of two 2D and define function spaces
nx = 16
ny = 16
#set initial time
t = 0
#set final time
t_f = 5
#set time step
#dt = 1.0
dt = 0.005
#calculate nt
nt = int(np.ceil(t_f/dt))
PETSc.Sys.Print('nt',nt)
#plot every n time steps
#nplot = 1
nplot = 50

####################################################################
#Subdomain 1
#the first subdomain will be split amongst processors
# this is set up ideally for subdomain 1 > subdomain 2
domain1 = mesh.create_rectangle(comm, [np.array([0, 0]), np.array([L, L])], [nx, nx], mesh.CellType.triangle)
V1 = fem.FunctionSpace(domain1, ("CG", 1))
u1 = ufl.TrialFunction(V1)
v1 = ufl.TestFunction(V1)
####################################################################
####################################################################
#Subdomain 2
#now we want entire second subdomain on EACH processor, so this will always be the smaller mesh
#MPI.COMM_SELF to not partition mesh
domain2 = mesh.create_rectangle(MPI.COMM_SELF, [np.array([0, 0]), np.array([L, L])], [ny, ny], mesh.CellType.triangle)
V2 = fem.FunctionSpace(domain2, ("CG", 1))
u2 = ufl.TrialFunction(V2)
v2 = ufl.TestFunction(V2)
###################################################################
###################################################################
#need local mass matrices to build global mass matrix
#mass of subdomain 1
m1 = u1*v1*ufl.dx
m1_form = fem.form(m1)
M1 = fem.petsc.assemble_matrix(m1_form)
M1.assemble()
#mass of subdomain 2
m2 = u2*v2*ufl.dx
m2_form = fem.form(m2)
M2 = fem.petsc.assemble_matrix(m2_form)
M2.assemble()
#################################################################
#################################################################
###################################################################
###################################################################
#save sizes of subdomain matrices
M1_sizes = M1.getLocalSize()
M1_global_size = M1.getSize() 
M2_sizes = M2.getLocalSize()
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
M=CFx.assemble.create_cartesian_mass_matrix(local_rows,global_rows,local_cols,global_cols)
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
#this gives ghost and owned dof coords
dof_coords1 = V1.tabulate_dof_coordinates()
dof_coords2 = V2.tabulate_dof_coordinates()
#suggested in forum, gives index of dofs I want
local_range1 = V1.dofmap.index_map.local_range
local_range2 = V2.dofmap.index_map.local_range
#vector of indexes that we want
dofs1 = np.arange(*local_range1,dtype=np.int32)
dofs2 = np.arange(*local_range2,dtype=np.int32)
#gets number of dofs owned
N_dof_1 = V1.dofmap.index_map.size_local
N_dof_2 = V2.dofmap.index_map.size_local
#hopefully the dof coordinates owned by the process
local_dof_coords1 = dof_coords1[0:N_dof_1,:domain1.topology.dim]
local_dof_coords2 = dof_coords2[0:N_dof_2,:domain2.topology.dim]

local_dof=CFx.transforms.cartesian_product_coords(local_dof_coords1,local_dof_coords2)

x = local_dof[:,0]
y = local_dof[:,1]
sigma = local_dof[:,2]
theta = local_dof[:,3]

#get global equation number of any node on entire global boundary
local_boundary_dofs = CFx.boundary.fetch_boundary_dofs(domain1,domain2,V1,V2,N_dof_1,N_dof_2)

#now only want subset that is the inflow, need to automate later
x_min = 0
y_min = 0
sigma_min = 0
theta_min = 0
dum1 = local_boundary_dofs[x[local_boundary_dofs]<=(x_min+1e-14)]
dum2 = local_boundary_dofs[y[local_boundary_dofs]<=(y_min+1e-14)]
dum3 = local_boundary_dofs[sigma[local_boundary_dofs]<=(sigma_min+1e-14)]
dum4 = local_boundary_dofs[theta[local_boundary_dofs]<=(theta_min+1e-14)]
local_boundary_dofs = np.unique(np.concatenate((dum1,dum2,dum3,dum4),0))
#local_boundary_dofs = dum2
global_boundary_dofs = local_boundary_dofs + local_range[0]
####################################################################
####################################################################
#generate any coefficients that depend on the degrees of freedom
c = 2*np.ones(local_dof.shape)
#c[:,2:] = 0
#c[:,1] = 1
#exact solution and dirichlet boundary
def u_func(x,y,sigma,theta,c,t):
    return np.sin(x-c[:,0]*t) + np.cos(y-c[:,1]*t) + np.sin(sigma-c[:,2]*t) + np.cos(theta-c[:,3]*t)
#####################################################################
#####################################################################
#Preallocate and load/assemble cartesian mass matrix!
#now need to mass matrixes for stiffness and RHS, also optionally can out put the nnz
M_NNZ = CFx.assemble.build_cartesian_mass_matrix(M1,M2,M1_sizes,M1_global_size,M2_sizes,M)
A.setPreallocationNNZ(M_NNZ)
##################################################################
##################################################################
#Loading A matrix routine
CFx.assemble.build_action_balance_stiffness(domain1,domain2,V1,V2,c,dt,A)
time_2 = time.time()
#if rank==0:
#    print(local_dof_coords1)
#    print(A.getValues(30,30))
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


B = F_dof.duplicate()
E = F_dof.duplicate()
L2_E = F_dof.duplicate()
u_cart = F_dof.duplicate()
u_exact = F_dof.duplicate()

E.setFromOptions()
B.setFromOptions()
L2_E.setFromOptions()
u_cart.setFromOptions()
u_exact.setFromOptions()

#if rank==0:
#    print('dof # ',str(30))
#    print(M.getValues(30,range(81)))
#    print(global_boundary_dofs)
#    print(local_dof[30,:])
#calculate pointwise values of RHS and put them in F_dof
#temp = u_true
#F_dof.setValues(rows,u_true)

#multiply F by Mass matrix to get B
#M.mult(F_dof,B)
#set Dirichlet boundary conditions
#B.setValues(global_boundary_dofs,u_d)
#print('local range')
#print(local_range)
#print('global boundary #')
#print(global_boundary_dofs)
#just want to test answer
#B.setValues(rows,u_2)
###################################################################
###################################################################
#Time step
#u_cart will hold solution
u_cart.setValues(rows,u_func(x,y,sigma,theta,c,t))
#u_cart.ghostUpdate()
u_cart.assemble()

#create a direct linear solver
#pc2 = PETSc.PC().create()
#this is a direct solve with lu
#pc2.setType('jacobi')
#pc2.setOperators(A)

ksp2 = PETSc.KSP().create() # creating a KSP object named ksp
ksp2.setOperators(A)
#ksp2.setType('cg')
#ksp2.setPC(pc2)

fname = 'ActionBalance_Propagation_CG/solution'
xdmf = io.XDMFFile(domain1.comm, fname+".xdmf", "w")
xdmf.write_mesh(domain1)


u = fem.Function(V1)
for i in range(nt):
    t+=dt
    u_2 = u_func(x,y,sigma,theta,c,t)
    u_d = u_2[local_boundary_dofs]
    #B = F_dof.duplicate()
    #B.setFromOptions()
    M.mult(u_cart,B)
    B.setValues(global_boundary_dofs,u_d)
    B.assemble()
    ksp2.solve(B, u_cart)
    #B.destroy()
    B.zeroEntries()

    # Save solution to file in VTK format
    if (i%nplot==0):
        u.vector.setValues(dofs1, np.array(u_cart.getArray()[4::N_dof_2]))
        xdmf.write_function(u, t)
        #hdf5_file.write(u,"solution",t)
xdmf.close()
time_end = time.time()
############################################################################
###########################################################################
#Post Processing section

#print whole solution
#print('Cartesian solution')
#print(u_cart.getArray()[:])
#print('Exact')
#print(u_true[:])

u_true = u_func(x,y,sigma,theta,c,t)
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
PETSc.Sys.Print("dof",(nx+1)**2*(ny+1)**2)
buildTime = time_2-time_start
solveTime = time_end-time_2
PETSc.Sys.Print(f'The build time is {buildTime} seconds')
PETSc.Sys.Print(f'The solve time is {solveTime} seconds')
