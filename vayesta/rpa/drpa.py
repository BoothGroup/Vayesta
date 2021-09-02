"""Straightforward N^6 implementation for dRPA in a basis set, based upon the standard Hermitian reformulation
used in TDHF approaches."""

import numpy as np

from pyscf import ao2mo

from vayesta.core.util import *
from timeit import default_timer as timer


class dRPA:
    """Approach based on equations expressed succinctly in the appendix of
    Furche, F. (2001). PRB, 64(19), 195120. https://doi.org/10.1103/PhysRevB.64.195120
    WARNING: Should only be used with canonical mean-field orbital coefficients in mf.mo_coeff and RHF.
    """

    def __init__(self, mf, log):
        self.mf = mf
        self.log = log

    @property
    def nocc(self):
        return sum(self.mf.mo_occ > 0)
    @property
    def nvir(self):
        return len(self.mf.mo_occ) - self.nocc
    @property
    def ov(self):
        return self.nocc * self.nvir

    def kernel(self):
        """Solve same-spin component of dRPA response.
        At level of dRPA this is the only contribution to correlation energy; introduction of exchange will lead to
        spin-flip contributions.
        """
        M, AmB, v = self._gen_arrays()
        e, c = np.linalg.eigh(M)
        self.freqs_ss = e ** (0.5)
        assert (all(e > 1e-12))
        self.ecorr = 0.5 * (sum(self.freqs_ss) - 2 * v.trace() - sum(AmB))

        XpY = np.einsum("n,p,pn->pn", self.freqs_ss ** (-0.5), AmB ** (0.5), c)
        XmY = np.einsum("n,p,pn->pn", self.freqs_ss ** (0.5), AmB ** (-0.5), c)
        self.XpY_ss = (XpY[:self.ov], XpY[self.ov:])
        self.XmY_ss = (XmY[:self.ov], XmY[self.ov:])

        self.freqs_sf = (AmB[:self.ov], AmB[self.ov:])
        return self.ecorr


    def _gen_arrays(self):
        # Only have diagonal components in canonical basis.
        eps = np.zeros((self.nocc, self.nvir))
        eps = eps + self.mf.mo_energy[self.nocc:]
        eps = (eps.T - self.mf.mo_energy[:self.nocc]).T
        eps = eps.reshape((self.ov,))

        AmB = np.concatenate([eps, eps])

        eris = self.ao2mo()
        # Get coulomb interaction in occupied-virtual space.
        v = eris[:self.nocc, self.nocc:, :self.nocc, self.nocc:].reshape((self.ov,self.ov))

        M = np.zeros((self.ov*2, self.ov*2))
        M[:self.ov, :self.ov] = M[:self.ov, self.ov:] = M[self.ov:, :self.ov] = M[self.ov:, self.ov:] = 2 * v
        M[np.diag_indices_from(M)] += AmB

        M = np.einsum("p,pq,q->pq", AmB**(0.5), M, AmB**(0.5))
        return M, AmB, v

    @property
    def mo_coeff(self):
        return self.mf.mo_coeff

    @property
    def nao(self):
        return self.mf.mol.nao

    def ao2mo(self):
        """Get the ERIs in MO basis
        """

        t0 = timer()
        self.log.info("ERIs will be four centered")
        mo_coeff = self.mo_coeff
        self.eri = ao2mo.incore.full(self.mf._eri, mo_coeff, compact=False)
        self.eri = self.eri.reshape((self.nao,) * 4)
        self.log.timing("Time for AO->MO:  %s", time_string(timer() - t0))

        return self.eri
