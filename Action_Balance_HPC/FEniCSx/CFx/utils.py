import numpy as np
from dolfinx import geometry,mesh
import ufl


def station_data(stations,domain,f):
    #takes in a numpy array of points in 2 dimensions
    #array should be Nx2
    #domain is a fenics mesh object
    #f is a fenics function defined on the given mesh
    if stations.shape[1] >= 4:
        print("Warning, input stations is not of correct dimension!!!")

    #now transpose this to proper format
    points = np.zeros((stations.shape[0],3))
    points[:,:stations.shape[1]] = stations
    
    bb_tree = geometry.BoundingBoxTree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = geometry.compute_collisions(bb_tree, points)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    for i, point in enumerate(points):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])

    f_values = []
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    f_values = f.eval(points_on_proc, cells)

    return points_on_proc,f_values

def gather_station(comm,root,local_stats,local_vals):
    rank = comm.Get_rank()
    #PETSc.Sys.Print("Trying to mpi gather")
    gathered_coords = comm.gather(local_stats,root=root)
    gathered_vals = comm.gather(local_vals,root=root)
    #PETSc.Sys.Print("directly after mpi gather",gathered_coords,gathered_vals)
    #PETSc.Sys.Print("size of new list",gathered_coords,gathered_vals)
    coords=[]
    vals = []
    if rank == root:
        for a in gathered_coords:
            if a.shape[0] != 0:
                for row in a:
                    coords.append(row)
        coords = np.array(coords)
        coords,ind1 = np.unique(coords,axis=0,return_index=True)
        
        for n in gathered_vals:
            if n.shape[0] !=0:
                for row in n:
                    vals.append(np.array(row))
        vals = np.array(vals)
        vals = vals[ind1]
    return coords,vals
    #PETSc.Sys.Print('station coords',coords)
    #PETSc.Sys.Print('station vals',vals)

def fix_diag(A,local_start,rank):
    diag = A.getDiagonal()
    dry_dofs = np.where(diag.getArray()==0)[0]
    dry_dofs = np.array(dry_dofs,dtype=np.int32) + local_start
    #print('number of 0s found on diag on rank',rank)
    #print(dry_dofs.shape)
    #print(dry_dofs)
    
    #fill in and reset
    #diag.setValues(dry_dofs,np.ones(dry_dofs.shape))
    
    #fill in matrix
    #A.setDiagonal(diag)
    return dry_dofs

#read in an adcirc mesh and give a fenicsx mesh
def ADCIRC_mesh_gen(comm,file_path):
    #specify file path as a string, either absolute or relative to where script is run
    #only compatible for adcirc fort.14 format
    adcirc_mesh=open(file_path,'r')
    title=adcirc_mesh.readline()

    #NE number of elements, NP number of grid points
    NE,NP=adcirc_mesh.readline().split()
    NE=int(NE)
    NP=int(NP)

    #initiate data structures
    NODENUM=np.zeros(NP)
    LONS=np.zeros(NP)
    LATS=np.zeros(NP)
    DPS=np.zeros(NP)
    ELEMNUM=np.zeros(NE)
    NM = np.zeros((NE,3)) #stores connectivity at each element

    #read node information line by line
    for i in range(NP):
        NODENUM[i], LONS[i], LATS[i], DPS[i] = adcirc_mesh.readline().split()
    #read in connectivity
    for i in range(NE):
        ELEMNUM[i], DUM, NM[i,0],NM[i,1], NM[i,2]=adcirc_mesh.readline().split()

    #(we need to shift nodenum down by 1)
    ELEMNUM=ELEMNUM-1
    NM=NM-1
    NODENUM=NODENUM-1

    #close file
    adcirc_mesh.close()

    gdim, shape, degree = 2, "triangle", 1
    cell = ufl.Cell(shape, geometric_dimension=gdim)
    element = ufl.VectorElement("Lagrange", cell, degree)
    domain = ufl.Mesh(element)
    coords = np.array(list(zip(LONS,LATS)))
    
    domain1 = mesh.create_mesh(comm, NM, coords, domain)
    return domain1


