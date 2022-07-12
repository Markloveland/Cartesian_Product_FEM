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
import time
time_start = time.time()
#get MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()



#specify geographic domain
x_min = 0.0
x_max = 3950 #should be 4000 (need to truncate a bit due to problem at 0 elevation)
y_min = 0.0
y_max = 100.0
n_x = 100 #number of elements toward  coastline
n_y = 100 # number of elements parallel to coastline




#determine spectral domain
omega_min=0.25 #smallest rad. frequency (needs to be larger than 0)
omega_max = 2.0 #largest rad. frequency
theta_min = -np.pi/4
theta_max = np.pi/4
n_sigma = 30 #number of elements in frequncy which is dimension no. 0   
n_theta = 10 #number of elements in theta


#set initial time
t = 0
#set final time
t_f = 1000
#set time step
dt = 5.0
#calculate nt
nt = int(np.ceil(t_f/dt))
PETSc.Sys.Print('nt',nt)
#plot every n time steps
nplot = 50
####################################################################
#Subdomain 1
#the first subdomain will be split amongst processors
# this is set up ideally for subdomain 1 > subdomain 2
mesh1 = RectangleMesh(comm,Point(x_min,y_min),Point(x_max,y_max),n_x,n_y)
V1 = FunctionSpace(mesh1, 'P', 1)
u1 = TrialFunction(V1)
v1 = TestFunction(V1)
####################################################################
####################################################################
#Subdomain 2
#now we want entire second subdomain on EACH processor, so this will always be the smaller mesh
#MPI.COMM_SELF to not partition mesh
mesh2 =  RectangleMesh(MPI.COMM_SELF,Point(omega_min,theta_min),Point(omega_max,theta_max),n_sigma,n_theta)
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
N_dof_1 = dof_coordinates1.shape[0]   
N_dof_2 = dof_coordinates2.shape[0]   
local_dof=CF.cartesian_product_coords(dof_coordinates1,dof_coordinates2) #do I actually need to compute this?
x = local_dof[:,0]
y = local_dof[:,1]
sigma = local_dof[:,2]
theta = local_dof[:,3]
#get global equation number of any node on entire global boundary
local_boundary_dofs = CF.fetch_boundary_dofs(V1,V2,dof_coordinates1,dof_coordinates2)

#now only want subset that is the inflow, need to automate later
#for shoaling case it shoud only be the boundary at x_min
dum1 = local_boundary_dofs[x[local_boundary_dofs]<=(x_min+1e-14)]
#dum2 = local_boundary_dofs[y[local_boundary_dofs]<=(y_min+1e-14)]
#dum3 = local_boundary_dofs[sigma[local_boundary_dofs]<=(sigma_min+1e-14)]
#dum4 = local_boundary_dofs[theta[local_boundary_dofs]<=(theta_min+1e-14)]


local_boundary_dofs = np.unique(dum1)
#local_boundary_dofs = np.unique(np.concatenate((dum1,dum2,dum3,dum4),0))
#local_boundary_dofs = dum2
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


# specify initial water depth and currents
# maybe can offload this to input script at some point
depth = 20 - x/200
u = np.zeros(local_dof.shape[0])
v = np.zeros(local_dof.shape[0])
c = compute_wave_speeds(x,y,sigma,theta,depth,u,v,g=9.81)

#dirichlet boundary
def u_func(x,y,sigma,theta,c,t):
    return np.sin(x-c[:,0]*t) + np.cos(y-c[:,1]*t) + np.sin(sigma-c[:,2]*t) + np.cos(theta-c[:,3]*t)
    #need gaussian guy here
#initial condition
def u_init(x,y,sigma,theta,c):
    #need gaussian guy here
###################################################################
###################################################################
#Preallocate and load/assemble cartesian mass matrix!
#now need to mass matrixes for stiffness and RHS, also optionally can out put the nnz
M_NNZ = CF.build_cartesian_mass_matrix(M1_pet,M2_pet,M1_sizes,M1_global_size,M2_sizes,M)
A.setPreallocationNNZ(M_NNZ)
##################################################################
##################################################################
#Loading A matrix routine
CF.build_stiffness_varying_action_balance(mesh1,V1,mesh2,V2,c,N_dof_2,dt,A)
time_2 = time.time()
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
#temp = u_true
#F_dof.setValues(rows,u_true)

#now matrix vector multiply with M to get actual right hand side
B = F_dof.duplicate()
E = F_dof.duplicate()
L2_E = F_dof.duplicate()
E.setFromOptions()
B.setFromOptions()
L2_E.setFromOptions()
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
u_cart = B.duplicate()
u_cart.setValues(rows,u_func(x,y,sigma,theta,c,t))
u_cart.assemble()

#create a direct linear solver
#pc2 = PETSc.PC().create()
#this is a direct solve with lu
#pc2.setType('lu')
#pc2.setOperators(A)

ksp2 = PETSc.KSP().create() # creating a KSP object named ksp
ksp2.setOperators(A)
ksp2.setType('cg')
#ksp2.setPC(pc2)

fname = 'ActionBalance/solution'
#pvd doesnt seem to work with new paraview
vtkfile = File(fname+'.pvd')
#vtkfile << mesh1
#try xdmf
#too slow
#file1 = XDMFFile(comm,'ActionBalance_xmf/mesh1.xdmf') 
#file1.write(mesh1,encoding=file1.Encoding)
#file1.close()
#try hdf5
#hdf5 works but I would need to generate a special xmf file so that it can be read in
#hdf5_file = HDF5File (comm,fname+'.hdf5', "w")

u = Function(V1)
for i in range(nt):
    t+=dt
    u_2 = u_func(x,y,sigma,theta,c,t)
    u_d = u_2[local_boundary_dofs]
    B = F_dof.duplicate()
    B.setFromOptions()
    M.mult(u_cart,B)
    B.setValues(global_boundary_dofs,u_d)
    B.assemble()
    ksp2.solve(B, u_cart)
    B.destroy()

    # Save solution to file in VTK format
    if (i%nplot==0):
        u.vector()[:] = np.array(u_cart.getArray()[4::N_dof_2])
        vtkfile << u
        #hdf5_file.write(u,"solution",t)

time_end = time.time()
####################################################################
###################################################################
#Post Processing section

#print whole solution
#print('Cartesian solution')
#print(u_cart.getArray()[:])
#print('Exact')
#print(u_true[:])

u_true = u_func(x,y,sigma,theta,c,t)
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
PETSc.Sys.Print("dof",(nx+1)**2*(ny+1)**2)
buildTime = time_2-time_start
solveTime = time_end-time_2
print(f'The build time is {buildTime} seconds')
print(f'The solve time is {solveTime} seconds')
##################################################################
#################################################################
#If I can find integral parameter like Hs then this section
# could be helpful for visualization


# Save solution to file in VTK format
#u = Function(V1)
#u.vector()[:] = np.array(u_cart.getArray()[4::N_dof_2])
#vtkfile = File('ActionBalance/solution.pvd')
#vtkfile << u


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

