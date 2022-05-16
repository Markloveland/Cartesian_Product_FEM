from fenics import *
from ufl import nabla_div
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D
import petsc4py
petsc4py.init()
from petsc4py import PETSc
import time
import CartFuncs as CF


#Goal here is to set up full action balance equations for some simple tests in serial
#using PETSc, and with time stepping

#First test will be a domain that is tensor product of 2 intervals
#and a sin wave will be propagating through the domain

#Second test will be the shoaling case from ONR A21


#define test case number first:
test_case_no=1
p_type = 'P' #choose basis type, P or CG 
p_degree = 1 #choose basis degree
nx = 15 #select refinement in one direction (number of elements)
ny = 15 #and then in the other

#define necessary parameters relevant to test case to construct mesh
if test_case_no==1:
	#specify parameters related to test 1
	T = 5 # final time step
	T_i=0
	num_steps = 1000 # number of time steps
	dt = T / num_steps  # time step size
	x_0 = 0
	x_1 = 10 #length of domain in x
	y_0 = 0
	y_1 = 10 #length of domain in y


#create meshes, all test cases here will be products of 1-D intervals
#mesh in x will be constructed first, then y
mesh1 = IntervalMesh(nx,x_0,x_1)
mesh2 = IntervalMesh(ny,y_0,y_1)

#create function spaces on each submesh
V1 = FunctionSpace(mesh1,p_type,p_degree)
V2 = FunctionSpace(mesh2,p_type,p_degree)
#generate global coordinates
dof_coordinates1=V1.tabulate_dof_coordinates()
dof_coordinates2=V2.tabulate_dof_coordinates()
N_dof_1 = len(dof_coordinates1)
N_dof_2 = len(dof_coordinates2)
global_dof=CF.cartesian_product_coords(dof_coordinates1,dof_coordinates2)
x=global_dof[:,0]
y=global_dof[:,1]
#want to mark global inflow boundary (when dot(c,n)<0)
global_boundary_dofs = CF.fetch_boundary_dofs(V1,V2,dof_coordinates1,dof_coordinates2)

#define necessary things that depend on the mesh or functional setting
#this includes propagation speed, boundary segments, source term, and initial condition
if test_case_no==1:
	# define propogation speeds as a vector cx,cy this will need to be modified
	def c_x(x,y,t):
		return np.ones(len(x))
	def c_y(x,y,t):
		return np.ones(len(y))	

        #define incoming boundary condition(exact solution as well in this case) as a function
	#of time, speed, position
	#as well as source S
	def u_D(x,y,cx,cy,t):
		return np.sin(x-cx*t)+np.cos(y-cy*t)
	def S(x,y,cx,cy,t):
		return np.zeros(len(x))

	#function for obtaining the global equation numbers for incoming and outcoming boundaries
	#needs the two meshes and then velocity c, try to generalize
	#now manually for now select only left as incoming boundary
	dum1 = global_boundary_dofs[x[global_boundary_dofs]<=(1e-14)]
	dum2 = global_boundary_dofs[y[global_boundary_dofs]<=(1e-14)]
	global_boundary_dofs = np.unique(np.concatenate((dum1,dum2),0))


#test to make sure its right
print(global_boundary_dofs)
print(x[global_boundary_dofs])
print(y[global_boundary_dofs])

#define initial condition for weak form
#define boundary can be tricky (dirichlet on incoming boundary)
#u_n = interpolate(u_D, V)
