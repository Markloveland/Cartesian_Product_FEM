import numpy as np
from dolfinx import fem
import ufl


#computation of any wave params/sources will take place here
def compute_wave_speeds(x,y,sigma,theta,depth,u,v,g=9.81):
    #need a function to calculate wave speed (phase and group) and wavenumber
    #takes in degrees of freedom and computes wave speeds pointwise
    N_dof = len(sigma)
    c_out = np.zeros((N_dof,4))
    temp = np.zeros(N_dof)
    k = np.zeros(N_dof)
    #employ 3 different approximations depending on water depth
    WGD=np.sqrt(depth/g)*g
    SND=sigma*np.sqrt(depth/g)

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

    #store the c_g (group velocity) as temporary variables
    temp[shallow_range]=cg_shallow(WGD[shallow_range])
    temp[mid_range],k[mid_range]=cg_mid(SND[mid_range],g,depth[mid_range],sigma[mid_range])
    temp[deep_range]=cg_deep(g,sigma[deep_range])
    #save these values in c_out and multiply by appropriate angles
    c_out[:,0] = temp*np.cos(theta)
    c_out[:,1] = temp*np.sin(theta)


    #now calculate wavenumber k and store temporarily
    k[shallow_range]=SND[shallow_range]/depth[shallow_range]
    k[deep_range]=sigma[deep_range]**2/g

    #now calculate c_sigma and c_theta, these are a bit more tricky

    #for now assuming H is constant in time but can fix this later
    #need to use FEniCS to calculate this!
    dHdt=0.0
    dHdy = 0.0
    dudy = 0.0
    dvdx = 0.0  #might not have to be 0, well see
    dvdy = 0.0

    #calc gradient of H w.r.t. x
    #this is just forward euler but only works for fixed geometry
    #instead we'll hard code for this case
    dHdx=-1.0/200.0
    dudx=0.0

    #now calculate velocity vectors
    #c_sigma
    c_out[:,2] = k*sigma/(np.sinh(2*k*depth)) *(dHdt + u*dHdx + v*dHdy) - temp*k*(dudx)
    #c theta
    c_out[:,3] = sigma/(np.sinh(2*k*depth))*(dHdx*np.sin(theta)- dHdy*np.cos(theta)) + \
        dudx*np.cos(theta)*np.sin(theta) - dudy*(np.cos(theta)**2) + dvdx*(np.sin(theta)**2) \
        -dvdy*np.cos(theta)*np.sin(theta)
    return c_out



def calculate_HS(u_cart,V2,local_size1,local_size2,local_range2):
    HS_vec = np.zeros(local_size1)
    dum = fem.Function(V2)

    intf = fem.form(dum*ufl.dx)
    
    #vector of global indexes that we want
    dofs = np.arange(*local_range2,dtype=np.int32)

    
    for i in range(local_size1):
        indx = i*local_size2
        #note this will only work if we only have 2nd domain unpartitioned!!!
        #(missing ghost values)
        #try to set proper values
        dum.vector.setValues(dofs,  np.array(u_cart.getArray()[indx:indx+local_size2]))
        local_intf = fem.assemble_scalar(intf)
        HS_vec[i] = 4*np.sqrt(abs(local_intf))

    return HS_vec

def calculate_wetdry(domain, V, depth_func,is_wet, min_depth=0.05):
    #loop through elements
    dim=domain.topology.dim
    imap = domain.topology.index_map(dim)
    ghost_cells = imap.num_ghosts
    num_cells = imap.size_local + ghost_cells

    dat_arr = np.zeros(num_cells)
    for a in range(num_cells):
        ind = V.dofmap.cell_dofs(a)
        vals = depth_func.x.array[ind]
        depth = np.min(vals)
        #define proper values in the element
        input_ind = is_wet._V.dofmap.cell_dofs(a)
        if depth > min_depth:
            out = 1
        else:
            out = 0
        is_wet.x.array[input_ind] = out

    return 0

