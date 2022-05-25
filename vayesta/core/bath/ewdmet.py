import numpy as np

from vayesta.core.linalg import recursive_block_svd
from vayesta.core.util import *
from vayesta.core import spinalg

from .bath import FragmentBath

class EwDMET_Bath(FragmentBath):

    def __init__(self, fragment, dmet_bath, max_order, *args, ewdmet_threshold=None, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.dmet_bath = dmet_bath
        if ewdmet_threshold is None:
            ewdmet_threshold = dmet_bath.dmet_threshold
        self.ewdmet_threshold = ewdmet_threshold
        self.max_order = max_order
        # Results
        self.c_cluster_occ = None
        self.c_cluster_vir = None
        self.c_env_occ = None
        self.c_env_vir = None

    def get_fock(self, *args, **kwargs):
        return self.fragment.base.get_fock_for_bath(*args, **kwargs)

    def kernel(self):
        # EwDMET orbitals:
        c_ewdmet_occ, self.c_env_occ = self._make_cluster('occ')
        c_ewdmet_vir, self.c_env_vir = self._make_cluster('vir')
        # Add DMET cluster
        self.c_cluster_occ = spinalg.hstack_matrices(self.dmet_bath.c_cluster_occ, c_ewdmet_occ)
        self.c_cluster_vir = spinalg.hstack_matrices(self.dmet_bath.c_cluster_vir, c_ewdmet_vir)

    def _make_svd(self, c_env, max_order=None):
        if max_order is None:
            max_order = self.max_order
        if (c_env.shape[-1] == 0):
            return c_env, np.zeros(0), np.zeros(0)

        c_frag = self.fragment.c_frag
        n_frag = c_frag.shape[-1]

        mo = spinalg.hstack_matrices(c_frag, c_env)
        fock = self.get_fock()
        f_occ = dot(mo.T, fock, mo)
        r_svd, sv, orders = recursive_block_svd(f_occ, n=n_frag, tol=self.ewdmet_threshold, maxblock=max_order)
        c_svd = np.dot(c_env, r_svd)
        return c_svd, sv, orders

    def _make_cluster(self, kind):
        if kind == 'occ':
            c_env = self.dmet_bath.c_env_occ
        elif kind == 'vir':
            c_env = self.dmet_bath.c_env_vir
        else:
            raise ValueError()
        c_bath, sv, orders = self._make_svd(c_env)

        # Output
        for order in range(1, self.max_order+1):
            mask = (orders == order)
            if np.count_nonzero(mask) == 0:
                break
            if order == 1:
                self.log.debug("%s EwDMET bath:" % ("Occupied" if kind == 'occ' else "Virtual"))
            self.log.debug("  Order %2d:  singular values= %s", order, np.array2string(sv[mask]))

        nbath = np.count_nonzero(orders <= self.max_order)
        return np.hsplit(c_bath, [nbath])

    def get_occupied_bath(self, *args, **kwargs):
        """Inherited bath classes can overwrite this to return additional occupied bath orbitals."""
        nao = self.mol.nao_nr()
        return np.zeros((nao, 0)), self.c_env_occ

    def get_virtual_bath(self, *args, **kwargs):
        """Inherited bath classes can overwrite this to return additional virtual bath orbitals."""
        nao = self.mol.nao_nr()
        return np.zeros((nao, 0)), self.c_env_vir
