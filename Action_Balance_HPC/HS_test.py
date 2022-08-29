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



#CRITICAL  = 50, // errors that may lead to data corruption and suchlike
#ERROR     = 40, // things that go boom
#WARNING   = 30, // things that may go boom later
#INFO      = 20, // information of general interest
#PROGRESS  = 16, // what's happening (broadly)
#TRACE     = 13, // what's happening (in detail)
#DBG       = 10  // sundry
level = 50
set_log_level(level)


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

omega_min= 0.25 #smallest rad. frequency (needs to be larger than 0)
omega_max = 2.0 #largest rad. frequency
theta_min = -10/180*np.pi
theta_max = 10/180*np.pi
n_sigma = 20 #number of elements in frequncy which is dimension no. 0   
n_theta = 6 #number of elements in theta



mesh2 =  RectangleMesh(Point(omega_min,theta_min),Point(omega_max,theta_max),n_sigma,n_theta)
V2 = FunctionSpace(mesh2, 'P', 1)
u2 = TrialFunction(V2)
v2 = TestFunction(V2)
dof_coordinates2=V2.tabulate_dof_coordinates()
N_dof_2 = dof_coordinates2.shape[0]
sigma = dof_coordinates2[:,0]
theta = dof_coordinates2[:,1]


def u_func(x,y,sigma,theta,c,t):
    #takes in dof and paramters
    HS = 1
    F_std = 0.1
    F_peak = 0.1
    Dir_mean = 0.0 #mean direction in degrees
    Dir_rad = Dir_mean*np.pi/(180)
    Dir_exp = 500
    #returns vector with initial condition values at global DOF
    aux1 = HS**2/(16*np.sqrt(2*np.pi)*F_std)
    aux3 = 2*F_std**2
    tol=1e-14
    aux2 = (sigma - ( np.pi*2*F_peak ) )**2
    E = (x<tol)*aux1*np.exp(-aux2/aux3)
    CTOT = np.sqrt(0.5*Dir_exp/np.pi)/(1.0 - 0.25/Dir_exp)
    A_COS = np.cos(theta - Dir_rad)
    CDIR = (A_COS>0)*CTOT*np.maximum(A_COS**Dir_exp, 1.0e-10)
    
    return E*CDIR

def u_func_1D(x,sigma):
    #takes in dof and paramters
    HS = 1
    F_std = 0.1
    F_peak = 0.1
    Dir_mean = 0.0 #mean direction in degrees
    Dir_rad = Dir_mean*np.pi/(180)
    Dir_exp = 500
    #returns vector with initial condition values at global DOF
    aux1 = HS**2/(16*np.sqrt(2*np.pi)*F_std)
    aux3 = 2*F_std**2
    tol=1e-14
    aux2 = (sigma - np.pi*2*F_peak )**2
    E = (x<tol)*aux1*np.exp(-aux2/aux3)
    CTOT = np.sqrt(0.5*Dir_exp/np.pi)/(1.0 - 0.25/Dir_exp)
    A_COS = np.cos(theta - Dir_rad)
    CDIR = (A_COS>0)*CTOT*np.maximum(A_COS**Dir_exp, 1.0e-10)
    
    return E

def Gauss_IC(x,sigmas):
    F_std =0.1
    F_peak = 0.1
    HS = 1 
    #takes in dof and paramters
    #returns vector with initial condition values at global DOF
    aux1 = HS**2/(16*np.sqrt(2*np.pi)*F_std)
    aux3 = 2*F_std**2
    tol=1e-14
    aux2 = (sigmas - 2*np.pi*F_peak)**2
    E = (x<tol)*aux1*np.exp(-aux2/aux3)
    return E

u = Function(V2)
u.vector()[:] = np.array(u_func(0,0,sigma,theta,0,0))

#p=plot(u)
#plt.colorbar(p)
#plt.savefig('HS_test.png')

print('HS computed')
print(4*np.sqrt(assemble(u*dx)))


#1D test 
#mesh1D = IntervalMesh(n_sigma,omega_min,omega_max)
#V1 = FunctionSpace(mesh1D, 'P', 1)
#u1 = TrialFunction(V1)
#v1 = TestFunction(V1)
#dof_coordinates1=V1.tabulate_dof_coordinates()
#sigma = dof_coordinates1[:,0]

#u = Function(V1)
#u.vector()[:] = np.array(u_func_1D(0,sigma))

#print('HS computed 1D')
#print(4*np.sqrt(assemble(u*dx)))






PETSc.Sys.Print('Attempt at extracting value at a fixed point')
p = Point(1.0,0.0)


val = np.zeros(1)
val[0] = -99999
if mesh2.bounding_box_tree().compute_first_entity_collision(p) < mesh2.num_cells():
    val[0] = u(p)

global_max = np.zeros(1)
comm.Reduce(val, global_max, op=MPI.MAX)
if rank ==0:
    print('value1:')
    print(global_max[0])

cell, distance = mesh2.bounding_box_tree().compute_closest_entity(p)
u_eval = u(p) if distance < DOLFIN_EPS else None

# Gather the results on process 0
computed_u = comm.gather(u_eval, root=0)
computed_u = comm.bcast(computed_u, root=0)

if rank == 0:
    print('value2:')
    print(computed_u)

#PETSc.Sys.Print("my rank", rank, "my val",computed_u)

p = [Point(1.0,0.0),Point(1.0,0.1)]

def peval(f, x,comm):
    '''Parallel synced eval'''
    try:
        yloc = f(x)
    except RuntimeError:
        yloc = np.inf*np.ones(1)

    yglob = np.zeros_like(yloc)
    comm.Reduce(yloc, yglob, op=MPI.MIN)
    return yglob
ux = np.zeros(len(p))
i=0
for a in p:
    ux[i] = peval(u,a,comm)
    i+=1
if rank==0:
    print('len p')
    print(len(p))
    print('value')
    print(ux)
