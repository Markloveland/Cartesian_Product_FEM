from __future__ import print_function
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

#all functions necessary to run Test Case 4
#this expression requires sigma to be in second index
def Gauss_Expression_1D(F_peak,F_std,HS):
    aux1 = aux1 = HS**2/(16*np.sqrt(2*np.pi)*F_std)
    aux3 = 2*F_std**2
    #CTOT = np.sqrt(0.5*500/np.pi)/(1-0.25/500)
    E=Expression('aux1*exp(-pow(x[1]-2*pi*F_peak,2)/aux3)',
                degree=4,aux1=aux1,aux3=aux3,F_peak=F_peak)
    return E

def Gauss_Expression_IC(F_peak,F_std,HS):
    aux1 = aux1 = HS**2/(16*np.sqrt(2*np.pi)*F_std)
    aux3 = 2*F_std**2
    tol=1e-14
    #CTOT = np.sqrt(0.5*500/np.pi)/(1-0.25/500)
    E=Expression('x[0] < tol ? aux1*exp(-pow(x[1]-2*pi*F_peak,2)/aux3): 0',
                degree=4,aux1=aux1,aux3=aux3,F_peak=F_peak,tol=tol,t=0)
    return E

def Gauss_IC(F_peak,F_std,HS,x,sigmas):
    #takes in dof and paramters
    #returns vector with initial condition values at global DOF
    aux1 = HS**2/(16*np.sqrt(2*np.pi)*F_std)
    aux3 = 2*F_std**2
    tol=1e-14
    E = (x<tol)*aux1*np.exp(-(sigmas - 2*np.pi*F_peak)**2/aux3)
    return E


#an Expression class to evaluate u(x,y) on the 1D domain (range y, x fixed)
#maybe modify so input can determine if we want range y (fixed x) or range x (fixed y)
#this case want space to be x, sigma to be y
class my1DExpression(UserExpression):
    def __init__(self,u_2d,x,**kwargs):
        super().__init__(**kwargs)
        self.u_2d=u_2d
        self.x=x
        #Expression.__init__(self)
        self._vx= np.array([0.])
        self._pt_x= np.array([0.,0.])
    def eval(self, values, x):
        self._pt_x[0]= self.x
        self._pt_x[1]= x[0]
        self.u_2d.eval(self._vx,self._pt_x)
        values[0] = self._vx[0]
        
#need a function to calculate wave speed (phase and group) and wavenumber
def swan_calc_wavespeeds_k(sigmas,depths,g=9.81):
    #taken from Jessica's code
    cg_out=np.zeros(len(sigmas))
    k_out=np.zeros(len(sigmas))
    WGD=np.sqrt(depths/g)*g
    SND=sigmas*np.sqrt(depths/g)
    
    shallow_range=np.argwhere(SND<1e-6)
    mid_range=np.argwhere((SND<2.5)&(SND>=1e-6))
    deep_range=np.argwhere(SND>=2.5)
    
    def cg_mid(SND,g,depths,sigmas):
        SND2=SND*SND
        C=np.sqrt(g*depths/(SND2 +1/(1+0.666*SND2+.445*SND2**2 
                                     - 0.105*SND2**3 + 0.272*SND2**4)))
        KND=sigmas*depths/C
        
        FAC1=2*KND/np.sinh(2*KND)
        N=0.5*(1+FAC1)
        return N*C,sigmas/C
    def cg_deep(g,sigmas):
        return 0.5*g/sigmas
    def cg_shallow(WGD):
        return WGD

    cg_out[shallow_range]=cg_shallow(WGD[shallow_range])
    k_out[shallow_range]=SND[shallow_range]/depths[shallow_range]
    cg_out[mid_range],k_out[mid_range]=cg_mid(SND[mid_range],g,depths[mid_range],sigmas[mid_range])
    cg_out[deep_range]=cg_deep(g,sigmas[deep_range])
    k_out[deep_range]=sigmas[deep_range]**2/g

    return cg_out,k_out

def calc_c_sigma(sigmas,k,depths,c_g,currents,z_coords,loc_num):
    ##Inputs
    #sigmas - takes in a vector of sigma values (sigma coordinate at each d.o.f) at one point in physical spaxe
    #thetas
    #k - wavenumber corresponding to the given sigmas and thetas
    #c_g - corresponding group velocity
    #(note sigmas, thetas,k,c_g should all be same length which is number of d.o.f in one 2D slice)
    #depths - depths in meters of ALL physical coordinates (not just current one)
    #z_coords - unique coordinates in physical space (not just current one)
    #currents - water velocity at ALL physical coordinates
    #loc_num - the number of the physical point (starts at 0)
    #(note depths,z_coords,currents should all be same length which is the number of unique points in physical space)
    ##Outputs
    #outputs a vector same size as sigmas that is the c_sigma at all d.o.f in ONE 2D slice 
    #of the 3D domain corresponding to a single point in physical space for c_theta and c_sigma
    
    #for now assuming H is constant in time but can fix this later
    dHdt=0.0
    dHdy = 0.0
    dudy = 0.0
    dvdx = 0.0 #migh not have to be 0, well see
    dvdy = 0.0
    
    #also going to assume v is zero but maybe can change this later
    v=0.0
    u=0.0#currents[loc_num]
    H=depths#[loc_num]
    #calc gradient of H w.r.t. x
    #this is just forward euler but only works for fixed geometry
    #instead we'll hard code for this case
    dHdx=-1.0/200.0
    dudx=0.0
    '''
    if loc_num == 0:
        dHdx =(depths[loc_num+1]-H)/(z_coords[loc_num+1]-z_coords[loc_num])
        dudx = (currents[loc_num+1]-u)/(z_coords[loc_num+1]-z_coords[loc_num])
    elif loc_num == len(z_coords)-1:
        dHdx = (H-depths[loc_num-1])/(z_coords[loc_num]-z_coords[loc_num-1])
        dudx = (u-currents[loc_num-1])/(z_coords[loc_num]-z_coords[loc_num-1])
    else:
        dHdx = (depths[loc_num+1]-depths[loc_num-1])/(z_coords[loc_num+1]-z_coords[loc_num-1])
        dudx = (currents[loc_num+1]-currents[loc_num-1])/(z_coords[loc_num+1]-z_coords[loc_num-1])
    '''
    
    #now calculate velocity vectors
    c_sigma = k*sigmas/(np.sinh(2*k*H)) *(dHdt + u*dHdx + v*dHdy) - c_g*k*(dudx)
    #c_theta = sigmas/(np.sinh(2*k*H))*(dHdx*np.sin(thetas)- dHdy*np.cos(thetas)) + \
    #    dudx*np.cos(thetas)*np.sin(thetas) - dudy*(np.cos(thetas)**2) + dvdx*(np.sin(thetas)**2) \
    #    -dvdy*np.cos(thetas)*np.sin(thetas)
    return c_sigma #,c_theta

###
###
###
#all functions relating to Cartesian product
def cartesian_product_dofs(array1,array2):
    #gives cartesian product of 2 1d vectors
    dim1=len(array1)
    dim2=len(array2)
    out_arr=np.zeros((dim1*dim2,2))
    c=0
    for a in range(dim1):
        for b in range(dim2):
            out_arr[c,:]=[array1[a],array2[b]]
            c=c+1
    return out_arr

def cartesian_product_coords(array1,array2):
    #gives cartesian product of 2 vectors
    dim1,n1=array1.shape
    dim2,n2=array2.shape
    out_arr=np.zeros((dim1*dim2,n1+n2))
    c=0
    for a in range(dim1):
        for b in range(dim2):
            #print(array1[a,:])
            #print(array2[b,:])
            out_arr[c,:]=np.append(array1[a,:],array2[b,:])
            c=c+1
    return out_arr

def cartesian_form_to_kroneck_form(indeces, len_dim_2):
    #designed to take list of indexes in the cartesian form
    #translate to single form as kron product would
    #only for cartesian product between 2 spaces
    #expects matrix with n rows, 2 columns
    num_indeces=indeces.shape[0]
    out_arr=np.zeros(num_indeces)
    for n in range(num_indeces):
        out_arr[n] = indeces[n,0]*len_dim_2 + indeces[n,1]
    return out_arr

def spec_mult(c,K):
    len_dof=len(c)
    for i in range(len_dof):
        for j in range(len_dof):
            K[i,j] = (c[i]+c[j])/2*K[i,j]
    return K

def u_pointwise(x,y):
    #exact solution at each point
    #case 1
    u = np.exp(-x**2-y**2)
    #case 2 homogeneous boundaries
    #u = np.sin(x)*np.sin(y)
    return u
def S_pointwise(x,y):
    #exact S at each point
    #case 1
    S = np.exp(-(x**2 + y**2)) - 2*(x+y)*np.exp(-(x**2 + y**2))
    #case 2 homogeneous boundaries
    #S = np.sin(x)*np.sin(y) + np.cos(x)*np.sin(y) + np.sin(x)*np.cos(y)
    return S

def kron_PETSC(K1,K2):
    #takes in to AIJ sparse matrices and gives sparse KRON of K1,K2
    K1_size = K1.getSize()
    K2_size = K2.getSize()
    Big_K = PETSc.Mat().createAIJ(K1_size[0]*K2_size[0],K1_size[1]*K2_size[1])
    Big_K.setUp()
    K2 = K2.getValues(range(K2_size[0]),range(K2_size[1]))
    for i in range(K1_size[0]):
        for j in range(K1_size[1]):
            Big_K.setValues(range(K2_size[0]*i,K2_size[0]*(i+1)), range(K2_size[1]*j, K2_size[1]*(j+1)), K2*K1.getValues(i,j))
    #Big_K.assemble()
    return Big_K

def Mass_assemble_PETSC(K11,K12,K14,K21,K22,K24):
    #specific to this problem
    # all matrices must be same size
    #np.kron(K11,K21)-np.kron(K12,K21)-np.kron(K11,K22) + \
    #+np.kron(K14,K21) + \
    #+ np.kron(K11,K24)
    K_size = K11.getSize()
    Big_K = PETSc.Mat().createAIJ(K_size[0]*K_size[0],K_size[1]*K_size[1])
    Big_K.setUp()
    K21 = K21.getValues(range(K_size[0]),range(K_size[1]))
    K22 = K22.getValues(range(K_size[0]),range(K_size[1]))
    K24 = K24.getValues(range(K_size[0]),range(K_size[1]))
   
    for i in range(K_size[0]):
        for j in range(K_size[1]):
            Big_K.setValues(range(K_size[0]*i,K_size[0]*(i+1)), range(K_size[1]*j, K_size[1]*(j+1)), K21*(K11.getValues(i,j) - K12.getValues(i,j) + K14.getValues(i,j)) + (-K22+K24)*K11.getValues(i,j)   )
    #Big_K.assemble()
    return Big_K
