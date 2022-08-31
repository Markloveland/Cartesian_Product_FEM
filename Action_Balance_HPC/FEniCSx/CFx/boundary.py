import numpy as np
import CFx.transforms
from dolfinx import mesh,fem
###########
#Handles any boundary issues in the global system
###########

def fetch_boundary_dofs(mesh1,mesh2,V1,V2,local_size1,local_size2):
    #outputs a vector of equation numbers in global system that are on the global boundary
    #input, the two function spaces of the subdomains in the cartesian product


    # Create facet to cell connectivity required to determine boundary facets
    # of each subdomain
    #for mesh1
    tdim1 = mesh1.topology.dim
    fdim1 = tdim1 - 1
    mesh1.topology.create_connectivity(fdim1, tdim1)
    boundary_facets1 = mesh.exterior_facet_indices(mesh1.topology)

    #for mesh2
    tdim2 = mesh2.topology.dim
    fdim2 = tdim2 - 1
    mesh2.topology.create_connectivity(fdim2, tdim2)
    boundary_facets2 = mesh.exterior_facet_indices(mesh2.topology)

    #equation number of the individual boundaries
    boundary_dofs1 = fem.locate_dofs_topological(V1, fdim1, boundary_facets1)
    boundary_dofs2 = fem.locate_dofs_topological(V2, fdim2, boundary_facets2)
    
    #now create array of global boundary dof numbers
    global_boundary_dofs=np.empty((len(boundary_dofs1)*local_size2 + local_size1*len(boundary_dofs2),2))

    ctr=0
    for j in boundary_dofs1:
        global_boundary_dofs[ctr*local_size2:(ctr+1)*local_size2,:] = \
        CFx.transforms.cartesian_product_dofs(np.array([j]),np.arange(local_size2))
        ctr=ctr+1

    last_ind = (ctr)*local_size2


    for j in boundary_dofs2:
        global_boundary_dofs[last_ind:last_ind+local_size1,:] = \
        CFx.transforms.cartesian_product_dofs(np.arange(local_size1),np.array([j]))
        last_ind = last_ind+local_size1    
    
    #sorts and also eliminates duplicates of "corners"
    if len(global_boundary_dofs) !=0:
        global_boundary_dofs=np.unique(global_boundary_dofs,axis=0)

    #have cartesian product of dof at entire boundary (this form should be easy to get coordinates in if needed)
    #now need to convert to global system dof as the kron function does
    global_boundary_dofs=CFx.transforms.cartesian_form_to_kroneck_form(global_boundary_dofs, local_size2)
    global_boundary_dofs=global_boundary_dofs.astype("int32")
    return global_boundary_dofs
