import numpy as np
#import matplotlib.pyplot as plt
from mpi4py import MPI
from scipy import sparse as sp
from petsc4py import PETSc
from dolfinx import fem,mesh,io
import ufl
import time
import CFx.wave
import CFx.utils
import CFx.assemble
import CFx.transforms
import CFx.boundary

print('hi')

filename = 'meshes/shoaling_unstructured.xdmf'
encoding= io.XDMFFile.Encoding.HDF5
with io.XDMFFile(MPI.COMM_WORLD, filename, "r", encoding=encoding) as file:
    domain1 = file.read_mesh()

V = fem.FunctionSpace(domain1, ("P", 1))
dof_coords = V.tabulate_dof_coordinates()
print(dof_coords.shape)
print(dof_coords)

'''
#get MPI variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
#specify file path as a string, either absolute or relative to where script is run
#only compatible for adcirc fort.14 format
file_path='meshes/depth2.grd'
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

'''
#domain1 = CFx.utils.ADCIRC_mesh_gen(MPI.COMM_WORLD,'meshes/depth2.grd')
