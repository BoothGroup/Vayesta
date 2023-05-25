import numpy as np
from vayesta.core.linalg import recursive_block_svd
from vayesta.core.util import dot
from vayesta.core import spinalg
from vayesta.core.bath.bath import Bath


class EwDMET_Bath_RHF(Bath):

    def __init__(self, fragment, dmet_bath, occtype, *args, threshold=None, max_order=20, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.dmet_bath = dmet_bath
        if occtype not in ('occupied', 'virtual'):
            raise ValueError("Invalid occtype: %s" % occtype)
        self.occtype = occtype
        if threshold is None:
            threshold = dmet_bath.dmet_threshold
        self.threshold = threshold
        self.max_order = max_order
        # Results
        self.c_bath, self.sv, self.orders = self.kernel()

    def get_fock(self, *args, **kwargs):
        return self.fragment.base.get_fock_for_bath(*args, **kwargs)

    def kernel(self):
        c_bath, sv, orders = self._make_svd()
        # Output
        for order in range(1, self.max_order+1):
            mask = (orders == order)
            if np.count_nonzero(mask) == 0:
                break
            if order == 1:
                self.log.info("EwDMET bath:")
            self.log.info("  Order %2d:  singular values= %s", order, np.array2string(sv[mask]))
        return c_bath, sv, orders

    def _make_svd(self):
        c_env = self.c_env
        if (c_env.shape[-1] == 0):
            return c_env, np.zeros(0), np.zeros(0)
        c_frag = self.fragment.c_frag
        n_frag = c_frag.shape[-1]
        mo = spinalg.hstack_matrices(c_frag, c_env)
        fock = self.get_fock()
        f_occ = dot(mo.T, fock, mo)
        r_svd, sv, orders = recursive_block_svd(f_occ, n=n_frag, tol=self.threshold, maxblock=self.max_order)
        c_svd = np.dot(c_env, r_svd)
        return c_svd, sv, orders

    @property
    def c_env(self):
        if self.occtype == 'occupied':
            return self.dmet_bath.c_env_occ
        if self.occtype == 'virtual':
            return self.dmet_bath.c_env_vir

    def get_bath(self, order):
        nbath = np.count_nonzero(self.orders <= order)
        return np.hsplit(self.c_bath, [nbath])
