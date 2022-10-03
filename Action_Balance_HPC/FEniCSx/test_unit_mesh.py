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



gdim, shape, degree = 2, "triangle", 1
cell = ufl.Cell(shape, geometric_dimension=gdim)
domain = ufl.Mesh(ufl.VectorElement("Lagrange", cell, degree))
x = np.array([[0., 0., 0.], [0., 1., 0.], [1., 1., 0.], [1. , 0., 0.]])
cells = np.array([[0, 1, 2],[0,2,3]], dtype=np.int64)
domain1 = mesh.create_mesh(MPI.COMM_WORLD, cells, x[:, :gdim], domain)

