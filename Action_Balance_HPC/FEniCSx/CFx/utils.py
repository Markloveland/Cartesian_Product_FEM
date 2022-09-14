import numpy as np
from dolfinx import geometry

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

