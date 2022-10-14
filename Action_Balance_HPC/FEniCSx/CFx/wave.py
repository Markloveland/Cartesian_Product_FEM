import numpy as np
from dolfinx import fem,cpp
import ufl


def interpolate_L2(f,V):
    #takes in bathymetry assuming P1, currents as dolfinx functions and interpolates
    #derivatives back to P1
    #see if we can finesse using L2 projection
    #V = f._V
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V) 
    a = u*v* ufl.dx
    L = f*v*ufl.dx
    problem = fem.petsc.LinearProblem(a, L, petsc_options={"ksp_type": "gmres"})
    ux = problem.solve()
    ux.vector.ghostUpdate()
    #L2 = f.dx(1)*v*ufl.dx
    #problem = fem.petsc.LinearProblem(a,L2,petsc_options={"ksp_type":"gmres"})
    #uy = problem.solve()
    return ux

def compute_tau(domain1,domain2,c,N_dof_1,N_dof_2):
    #need to do this for domain1 and domain2

    #get h as elementwise
    tdim = domain1.topology.dim
    num_cells1 = domain1.topology.index_map(tdim).size_local
    h1 = cpp.mesh.h(domain1, tdim, range(num_cells1))
    #save as a DG function
    cellwise = fem.FunctionSpace(domain1, ("DG", 0))
    V1 = fem.FunctionSpace(domain1,("CG",1))
    height1 = fem.Function(cellwise)
    height1.x.array[:num_cells1] = h1
    height1.vector.ghostUpdate()
    h_p1 = interpolate_L2(height1,V1)
    #print('h_p1',height1.vector.getArray())
    #get h as elementwise
    tdim = domain2.topology.dim
    num_cells2 = domain2.topology.index_map(tdim).size_local
    h2 = cpp.mesh.h(domain2, tdim, range(num_cells2))
    #save as a DG function
    cellwise2 = fem.FunctionSpace(domain2, ("DG", 0))
    V2 = fem.FunctionSpace(domain2,("CG",1))
    height2 = fem.Function(cellwise2)
    height2.x.array[:num_cells2] = h2
    height2.vector.ghostUpdate()
    h_p2 = interpolate_L2(height2,V2)

    #using the 2 vectors of h, generate an estimate for h in global domain
    #get N_dof some how

    h_1 = np.kron(h_p1.vector.getArray(),np.ones(N_dof_2))
    h_2 = np.kron(np.ones(N_dof_1),h_p2.vector.getArray())

    h = np.sqrt(h_1**2+h_2**2)

    #now using pointwise c get a pointwise tau =h / |/c|/
    tau_pointwise = h/np.sqrt((c**2).sum(axis=1))

    return tau_pointwise

def compute_tau_old(mesh1,mesh2,c_vals,subdomain=0):
    #hardcoded for uniform mesh this time
    tdim = mesh1.topology.dim
    num_cells = mesh1.topology.index_map(tdim).size_local
    h1 = cpp.mesh.h(mesh1, tdim, range(num_cells)).max()
    #print('h1max',h1)
    tdim = mesh2.topology.dim
    num_cells = mesh2.topology.index_map(tdim).size_local
    h2 = cpp.mesh.h(mesh2, tdim, range(num_cells)).max()

    h = np.sqrt(h1**2+h2**2)
    temp = c_vals**2
    c_mag = np.sqrt(temp.sum(axis=1))
    #tau = h / /|c/|
    tau_points = np.array(h/c_mag)
    return tau_points


#computation of any wave params/sources will take place here
def compute_wave_speeds_pointwise(x,y,sigma,theta,depth,u,v,dHdx=-1.0/200,dHdy=0.0,g=9.81):
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
    #dHdy = 0#-1.0/200#0.0
    dudy = 0.0
    dvdx = 0.0  #might not have to be 0, well see
    dvdy = 0.0

    #calc gradient of H w.r.t. x
    #this is just forward euler but only works for fixed geometry
    #instead we'll hard code for this case
    #dHdx=-1.0/200.0#0
    dudx=0.0

    #now calculate velocity vectors
    #c_sigma
    c_out[:,2] = k*sigma/(np.sinh(2*k*depth)) *(dHdt + u*dHdx + v*dHdy) - temp*k*(dudx)
    #c theta
    c_out[:,3] = sigma/(np.sinh(2*k*depth))*(dHdx*np.sin(theta)- dHdy*np.cos(theta)) + \
        dudx*np.cos(theta)*np.sin(theta) - dudy*(np.cos(theta)**2) + dvdx*(np.sin(theta)**2) \
        -dvdy*np.cos(theta)*np.sin(theta)
    return c_out



#computation of any wave params/sources will take place here
def compute_wave_speeds(x,y,sigma,theta,depth_func,u_func,v_func,N_dof_2,g=9.81, min_depth = 0.05):
    dHdx_func = interpolate_L2(depth_func.dx(0),depth_func._V)
    dHdy_func = interpolate_L2(depth_func.dx(1),depth_func._V)
    
    #dHdx_func,dHdy_func = interpolate_gradients(depth_func)    

    #compute depth at all dof
    depth = np.kron(depth_func.vector.getArray(),np.ones(N_dof_2))
    u = np.kron(u_func.vector.getArray(),np.ones(N_dof_2))
    v = np.kron(v_func.vector.getArray(),np.ones(N_dof_2))
    dHdx = np.kron(dHdx_func.vector.getArray(),np.ones(N_dof_2))
    dHdy = np.kron(dHdy_func.vector.getArray(),np.ones(N_dof_2))

    #need a function to calculate wave speed (phase and group) and wavenumber
    #takes in degrees of freedom and computes wave speeds pointwise
    N_dof = len(sigma)
    c_out = np.ones((N_dof,4))
    

    dry_dofs_local = np.array(np.where(depth<min_depth)[0],dtype=np.int32)
    ##dry_dofs = dry_dofs_local + local_range[0]
    wet_dofs_local = np.where(depth>=min_depth)[0] 
    

    temp = np.zeros(wet_dofs_local.shape)
    k = np.zeros(wet_dofs_local.shape)


    c_out[wet_dofs_local,:] = compute_wave_speeds_pointwise(x[wet_dofs_local],y[wet_dofs_local],sigma[wet_dofs_local],theta[wet_dofs_local],depth[wet_dofs_local],
            u[wet_dofs_local],v[wet_dofs_local],dHdx=dHdx[wet_dofs_local],dHdy=dHdy[wet_dofs_local])
    '''
    #employ 3 different approximations depending on water depth
    WGD=np.sqrt(depth_wet/g)*g
    SND=sigma_wet*np.sqrt(depth_wet/g)

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
    temp[mid_range],k[mid_range]=cg_mid(SND[mid_range],g,depth_wet[mid_range],sigma_wet[mid_range])
    temp[deep_range]=cg_deep(g,sigma_wet[deep_range])
    #save these values in c_out and multiply by appropriate angles
    c_out[wet_dofs_local,0] = temp*np.cos(theta_wet)
    c_out[wet_dofs_local,1] = temp*np.sin(theta_wet)


    #now calculate wavenumber k and store temporarily
    k[shallow_range]=SND[shallow_range]/depth_wet[shallow_range]
    k[deep_range]=sigma_wet[deep_range]**2/g

    #now calculate c_sigma and c_theta, these are a bit more tricky

    #for now assuming H is constant in time but can fix this later
    #need to use FEniCS to calculate this!
    dHdt=0.0
    #dHdy = 0#-1.0/200#0.0
    dudy = 0.0
    dvdx = 0.0  #might not have to be 0, well see
    dvdy = 0.0

    #calc gradient of H w.r.t. x
    #this is just forward euler but only works for fixed geometry
    #instead we'll hard code for this case
    #dHdx=-1.0/200.0#0
    dudx=0.0

    #now calculate velocity vectors
    #c_sigma
    c_out[wet_dofs_local,2] = k*sigma_wet/(np.sinh(2*k*depth_wet)) *(dHdt + u_wet*dHdx_wet + v_wet*dHdy_wet) - temp*k*(dudx)
    #c theta
    c_out[wet_dofs_local,3] = sigma_wet/(np.sinh(2*k*depth_wet))*(dHdx_wet*np.sin(theta_wet)- dHdy_wet*np.cos(theta_wet)) + \
        dudx*np.cos(theta_wet)*np.sin(theta_wet) - dudy*(np.cos(theta_wet)**2) + dvdx*(np.sin(theta_wet)**2) \
        -dvdy*np.cos(theta_wet)*np.sin(theta_wet)
    '''
    return c_out,dry_dofs_local



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

