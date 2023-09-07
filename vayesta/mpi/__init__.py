from vayesta.mpi.interface import MPI_Interface
from vayesta.mpi.rma import RMA_Dict

mpi = None


def init_mpi(use_mpi, required=True):
    global mpi
    if use_mpi:
        mpi = MPI_Interface("mpi4py", required=required)
    else:
        mpi = MPI_Interface(None)
    return mpi
