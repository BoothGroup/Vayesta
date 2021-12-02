import functools
import logging

import vayesta
import vayesta.core
from vayesta.core.mpi import mpi
from vayesta.core.util import *


log = logging.getLogger(__name__)


def scf_with_mpi(mf, mpi_rank=0):
    """Use to run SCF only on the master node and broadcast result afterwards."""

    if not mpi:
        return mf

    bcast = functools.partial(mpi.world.bcast, root=mpi_rank)
    kernel_orig = mf.kernel

    def mpi_kernel(self, *args, **kwargs):
        if mpi.rank == mpi_rank:
            res = kernel_orig(*args, **kwargs)
        else:
            res = None
            # Generate auxiliary cell, compensation basis etc,..., but not 3c integrals:
            if hasattr(self, 'with_df') and self.with_df.auxcell is None:
                self.with_df.build(with_j3c=False)

        # Broadcast results
        with log_time(log.timing, "Time for MPI broadcast of SCF results: %s"):
            res = bcast(res)
            if hasattr(self, 'with_df'):
                self.with_df._cderi = bcast(self.with_df._cderi)
            self.converged = bcast(self.converged)
            self.e_tot = bcast(self.e_tot)
            self.mo_energy = bcast(self.mo_energy)
            self.mo_occ = bcast(self.mo_occ)
            self.mo_coeff = bcast(self.mo_coeff)
        return res

    mf.kernel = mpi_kernel.__get__(mf)

    return mf


if __name__ == '__main__':

    import numpy as np

    import pyscf
    import pyscf.pbc
    import pyscf.pbc.gto

    mol = pyscf.pbc.gto.Cell()
    mol.a = 3*np.eye(3)
    mol.atom = 'He 0 0 0'
    mol.basis = 'cc-pVDZ'
    mol.build()

    mf = pyscf.pbc.scf.RHF(mol)
    mf = mf.density_fit(auxbasis='cc-pVDZ-ri')
    mf = scf_with_mpi(mf)
    mf.kernel()

    print("MPI rank = %d e_tot= %.12e" % (mpi.rank, mf.e_tot))
