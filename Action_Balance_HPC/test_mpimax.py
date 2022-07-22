from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

x =np.zeros(1)
x[0] = -9999
if rank==2:
    x[0] = 5
global_max = np.zeros(1)
comm.Reduce(x, global_max, op=MPI.MAX)
print("my rank", rank, "my global val",global_max[0])

