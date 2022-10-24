
"""
Action Balance Equation Solver
This algorithm for loading will be same required of Action Balance Equation
    du/dt + \/.cu = f
Case from ONR Testbed case A21
"""

import numpy as np
#import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy import sparse as sp
from petsc4py import PETSc
from dolfinx import fem,mesh,io
import ufl
import time
import CFx.utils
import CFx.assemble
import CFx.transforms
import CFx.boundary
import CFx.wave
time_start = time.time()



#get MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

#specify bouns in geographic mesh
x_min = 0.0
x_max = 20000
y_min = 0.0
y_max = 4000
# Create cartesian mesh of two 2D and define function spaces
nx = 100
ny = 100
# define spectral domain
omega_min = 0.25
omega_max = 2.0
theta_min = np.pi/2 - 10/180*np.pi
theta_max = np.pi/2 + 10/180*np.pi
n_sigma = 30
n_theta = 24
#set initial time
t = 0
#set final time
t_f = 2000/2
#set time step
dt = 0.5


#Subdomain 1
#the first subdomain will be split amongst processors
# this is set up ideally for subdomain 1 > subdomain 2
#domain1 = mesh.create_rectangle(comm, [np.array([x_min, y_min]), np.array([x_max, y_max])], [nx, ny], mesh.CellType.triangle)
filename = 'meshes/shoaling_unstructured.xdmf'
encoding= io.XDMFFile.Encoding.HDF5
with io.XDMFFile(MPI.COMM_WORLD, filename, "r", encoding=encoding) as file:
    domain1 = file.read_mesh()


V1 = fem.FunctionSpace(domain1, ("CG", 1))
u1 = ufl.TrialFunction(V1)
v1 = ufl.TestFunction(V1)


dof_coords1 = V1.tabulate_dof_coordinates()
#suggested in forum, gives index of dofs I want
local_range1 = V1.dofmap.index_map.local_range
#vector of indexes that we want
dofs1 = np.arange(*local_range1,dtype=np.int32)
#gets number of dofs owned
N_dof_1 = V1.dofmap.index_map.size_local
#hopefully the dof coordinates owned by the process
local_dof_coords1 = dof_coords1[0:N_dof_1,:domain1.topology.dim]
#for now lets set depth as x coordinate itself
#eventually this maybe read in from txt file or shallow water model


#let's see if we can just assign some normal function to it
HS = fem.Function(V1)
#HS_vec = CFx.wave.calculate_HS(u_cart,V2,N_dof_1,N_dof_2,local_range2)
#depths = 20 - local_dof_coords1[:,1]/200
#HS.x.array[:] = 20 - dof_coords1[:,1]/200
def u_func(x,y,sigma,theta,c,t):
    #takes in dof and paramters
    HS = 1
    F_std = 0.1
    F_peak = 0.1
    Dir_mean = 90.0 #mean direction in degrees
    Dir_rad = Dir_mean*np.pi/(180)
    Dir_exp = 500
    #returns vector with initial condition values at global DOF
    aux1 = HS**2/(16*np.sqrt(2*np.pi)*F_std)
    aux3 = 2*F_std**2
    tol=1e-11
    aux2 = (sigma - ( np.pi*2*F_peak ) )**2
    E = (y<tol)*np.ones(y.shape)#aux1*np.exp(-aux2/aux3)
    #CTOT = np.sqrt(0.5*Dir_exp/np.pi)/(1.0 - 0.25/Dir_exp)
    #A_COS = np.cos(theta - Dir_rad)
    #CDIR = (A_COS>0)*CTOT*np.maximum(A_COS**Dir_exp, 1.0e-10)
    return E#*CDIR


#HS.x.array[:] = u_func(0,dof_coords[:,1],0,0,0,0)
HS_local = u_func(0,local_dof_coords1[:,1],0,0,0,0)
HS.vector.setValues(dofs1,np.array(HS_local))
HS.vector.ghostUpdate()
fname = 'unstrucrtured_debug_2/solution'
xdmf = io.XDMFFile(domain1.comm, fname+".xdmf", "w")
xdmf.write_mesh(domain1)
xdmf.write_function(HS)
xdmf.close()
