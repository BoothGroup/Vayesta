import numpy as np

from vayesta.core.linalg import recursive_block_svd
from vayesta.core.util import *

from .bath import FragmentBath

class EwDMET_Bath(FragmentBath):

    def __init__(self, fragment, dmet_bath, *args, ewdmet_threshold=None, kmax=10, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.dmet_bath = dmet_bath
        if ewdmet_threshold is None:
            ewdmet_threshold = dmet_bath.dmet_threshold
        self.ewdmet_threshold = ewdmet_threshold
        self.kmax = kmax
        # Results
        # Bath orbital coefficients:
        self.c_occ = None
        self.c_vir = None
        # Bath orbital orders:
        self.k_occ = None
        self.k_vir = None
        # Bath orbital singular values:
        self.sv_occ = None
        self.sv_vir = None


    @property
    def c_cluster_occ(self):
        """Occupied DMET cluster orbitals."""
        return self.dmet_bath.c_cluster_occ

    @property
    def c_cluster_vir(self):
        """Virtual DMET cluster orbitals."""
        return self.dmet_bath.c_cluster_vir

    def get_fock(self, *args, **kwargs):
        return self.fragment.base.get_fock_for_bath(*args, **kwargs)

    def kernel(self):
        self.c_occ, self.sv_occ, self.k_occ = self.make_svd(self.dmet_bath.c_env_occ)
        self.c_vir, self.sv_vir, self.k_vir = self.make_svd(self.dmet_bath.c_env_vir)
        for k in range(1, self.kmax+1):
            kmask = (k == self.k_occ)
            if np.count_nonzero(kmask) == 0:
                break
            if k == 1:
                self.log.debug("Occupied EwDMET bath:")
            self.log.debug("  Order %2d:  singular values= %s", k, np.array2string(self.sv_occ[kmask]))
        for k in range(1, self.kmax+1):
            kmask = (k == self.k_vir)
            if np.count_nonzero(kmask) == 0:
                break
            if k == 1:
                self.log.debug("Virtual EwDMET bath:")
            self.log.debug("  Order %2d:  singular values= %s", k, np.array2string(self.sv_vir[kmask]))

    def make_svd(self, c_env):
        if (c_env.shape[-1] == 0):
            return c_env, np.zeros(0), np.zeros(0)

        c_frag = self.fragment.c_frag
        n_frag = c_frag.shape[-1]

        mo = stack_mo(c_frag, c_env)
        fock = self.get_fock()
        f_occ = dot(mo.T, fock, mo)
        r_svd, sv, orders = recursive_block_svd(f_occ, n=n_frag, tol=self.ewdmet_threshold, maxblock=self.kmax)
        c_svd = np.dot(c_env, r_svd)
        return c_svd, sv, orders

    def get_occupied_bath(self, kmax, **kwargs):
        nbath = np.count_nonzero(self.k_occ <= kmax)
        return np.hsplit(self.c_occ, [nbath])

    def get_virtual_bath(self, kmax, **kwargs):
        nbath = np.count_nonzero(self.k_vir <= kmax)
        return np.hsplit(self.c_vir, [nbath])

# TODO
