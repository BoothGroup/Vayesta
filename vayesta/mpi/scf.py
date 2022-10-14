import functools
import logging
import pyscf
import pyscf.df
import vayesta
import vayesta.core
from vayesta.core.util import *


def scf_with_mpi(mpi, mf, mpi_rank=0, log=None):
    """Use to run SCF only on the master node and broadcast result afterwards."""

    if not mpi:
        return mf

    bcast = functools.partial(mpi.world.bcast, root=mpi_rank)
    kernel_orig = mf.kernel
    log = log or mpi.log or logging.getLogger(__name__)

    def mpi_kernel(self, *args, **kwargs):
        df = getattr(self, 'with_df', None)
        if mpi.rank == mpi_rank:
            log.info("MPI rank= %3d is running SCF", mpi.rank)
            with log_time(log.timing, "Time for SCF: %s"):
                res = kernel_orig(*args, **kwargs)
            log.info("MPI rank= %3d finished SCF", mpi.rank)
        else:
            res = None
            # Generate auxiliary cell, compensation basis etc,..., but not 3c integrals:
            if df is not None:
                # Molecules
                if getattr(df, 'auxmol', False) is None:
                    df.auxmol = pyscf.df.addons.make_auxmol(df.mol, df.auxbasis)
                # Solids
                elif getattr(df, 'auxcell', False) is None:
                    df.build(with_j3c=False)
            log.info("MPI rank= %3d is waiting for SCF results", mpi.rank)
        mpi.world.barrier()

        # Broadcast results
        with log_time(log.timing, "Time for MPI broadcast of SCF results: %s"):
            res = bcast(res)
            if df is not None:
                df._cderi = bcast(df._cderi)
            self.converged = bcast(self.converged)
            self.e_tot = bcast(self.e_tot)
            self.mo_energy = bcast(self.mo_energy)
            self.mo_occ = bcast(self.mo_occ)
            self.mo_coeff = bcast(self.mo_coeff)
        return res

    mf.kernel = mpi_kernel.__get__(mf)
    mf.with_mpi = True

    return mf
