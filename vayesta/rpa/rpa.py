"""Straightforward N^6 implementation for RPA in a finite basis set with arbitrary interaction kernel, based upon the
standard Hermitian reformulation used in TDHF approaches.
Note that we only use the spin-block formulation of all matrices, rather than full spin-adaptation, which means our
final diagonalisation is 2^3=8 times more expensive than hypothetically possible. However, this code is only for
comparison."""

import logging
from timeit import default_timer as timer

import numpy as np
import scipy.linalg

from pyscf import ao2mo
from vayesta.core.util import *


class RPA:
    """Approach based on equations expressed succinctly in the appendix of
    Furche, F. (2001). PRB, 64(19), 195120. https://doi.org/10.1103/PhysRevB.64.195120
    WARNING: Should only be used with canonical mean-field orbital coefficients in mf.mo_coeff and RHF.
    """

    def __init__(self, mf, log=None):
        self.mf = mf
        self.log = log or logging.getLogger(__name__)

    @property
    def nocc(self):
        return sum(self.mf.mo_occ > 0)

    @property
    def nvir(self):
        return len(self.mf.mo_occ) - self.nocc

    @property
    def ov(self):
        return self.nocc * self.nvir

    @property
    def e_corr(self):
        try:
            return self.e_corr_ss + self.e_corr_sf
        except AttributeError as e:
            self.log.critical("Can only access rpa.e_corr after running rpa.kernel.")

    @property
    def e_tot(self):
        return self.mf.e_tot + self.e_corr

    def kernel(self, xc_kernel="rpax"):
        """Solve for RPA response; solve same-spin (ss) and spin-flip (sf) separately.
        If doing dRPA spin-flip is trivial, so for large calculations use dRPA specific
        """
        t_start = timer()

        ApB_ss, AmB_ss, ApB_sf, AmB_sf = self._build_arrays(xc_kernel)

        def solve_RPA_problem(ApB, AmB):
            AmB_rt = scipy.linalg.sqrtm(AmB)
            M = np.linalg.multi_dot([AmB_rt, ApB, AmB_rt])
            e, c = np.linalg.eigh(M)
            freqs = e**0.5
            assert all(e > 1e-12)
            ecorr_contrib = 0.5 * (sum(freqs) - 0.5 * (ApB.trace() + AmB.trace()))
            XpY = np.einsum("n,pn->pn", freqs ** (-0.5), np.dot(AmB_rt, c))
            XmY = np.einsum(
                "n,pn->pn", freqs ** (0.5), np.dot(np.linalg.inv(AmB_rt), c)
            )
            return (
                freqs,
                ecorr_contrib,
                (XpY[: self.ov], XpY[self.ov :]),
                (XmY[: self.ov], XmY[self.ov :]),
            )

        t0 = timer()
        self.freqs_ss, self.e_corr_ss, self.XpY_ss, self.XmY_ss = solve_RPA_problem(
            ApB_ss, AmB_ss
        )
        self.freqs_sf, self.e_corr_sf, self.XpY_sf, self.XmY_sf = solve_RPA_problem(
            ApB_sf, AmB_sf
        )
        self.log.timing("Time to solve RPA problems: %s", time_string(timer() - t0))

        if xc_kernel == "rpax":
            # Additional factor of 0.5.
            self.e_corr_ss *= 0.5
            self.e_corr_sf *= 0.5

        self.log.info("Total RPA wall time:  %s", time_string(timer() - t_start))

        return self.e_corr

    def _build_arrays(self, xc_kernel="rpax"):
        t0 = timer()
        # Only have diagonal components in canonical basis.
        eps = np.zeros((self.nocc, self.nvir))
        eps = eps + self.mf.mo_energy[self.nocc :]
        eps = (eps.T - self.mf.mo_energy[: self.nocc]).T
        eps = eps.reshape((self.ov,))
        # Get interaction kernel
        (k_pss, k_mss, k_psf, k_msf) = self.get_interaction_kernel(xc_kernel)

        def combine_spin_components(k1, k2):
            res = np.zeros((2 * self.ov, 2 * self.ov))
            res[: self.ov, : self.ov] = res[self.ov :, self.ov :] = k1
            res[: self.ov, self.ov :] = res[self.ov :, : self.ov] = k2
            return res

        ApB_ss = combine_spin_components(*k_pss)
        AmB_ss = combine_spin_components(*k_mss)
        ApB_sf = combine_spin_components(*k_psf)
        AmB_sf = combine_spin_components(*k_msf)

        # Construct full irreducible polarisability, then add in to diagonal.
        fulleps = np.concatenate([eps, eps])
        ix_diag = np.diag_indices(2 * self.ov)
        ApB_ss[ix_diag] += fulleps
        ApB_sf[ix_diag] += fulleps
        AmB_ss[ix_diag] += fulleps
        AmB_sf[ix_diag] += fulleps
        self.log.timing("Time to build RPA arrays: %s", time_string(timer() - t0))

        return ApB_ss, AmB_ss, ApB_sf, AmB_sf

    def get_interaction_kernel(self, xc_kernel="rpax", tda=False):
        """Construct the required components of the interaction kernel, separated into same-spin and spin-flip
        components, as well as spin contributions for A+B and A-B.
        The results is a length-4 tuple, giving the spin components of respectively
            (ss K_(A+B), ss K_(A-B), sf K_(A+B), sf K_(A-B)).
        In RHF both contributions both only have two distinct spin components, so there are a total of
        8 distinct spatial kernels for a general interaction.
        For spin contributions we use the orderings
            -(aaaa, aabb) for ss contributions.
            -(abab, abba) for st contributions (ie, whether the particle states have the
                same spin in both pairs or not). Sorry for clunky description...

        If TDA is specified all appropriate couplings will be zeroed.
        """
        if xc_kernel is None or xc_kernel.lower() == "drpa":
            self.log.info("RPA using coulomb interaction kernel.")
            eris = self.ao2mo()

            v = eris[: self.nocc, self.nocc :, : self.nocc, self.nocc :].reshape(
                (self.ov, self.ov)
            )
            # Only nonzero contribution is between same-spin excitations due to coulomb interaction.
            kernel = (
                (2 * v, 2 * v),
                (np.zeros_like(v), np.zeros_like(v)),
                (np.zeros_like(v), np.zeros_like(v)),
                (np.zeros_like(v), np.zeros_like(v)),
            )

        elif xc_kernel.lower() == "rpax":
            self.log.info("RPA using coulomb-exchange interaction kernel.")
            eris = self.ao2mo()
            v = eris[: self.nocc, self.nocc :, : self.nocc, self.nocc :]
            ka = np.einsum(
                "ijab->iajb", eris[: self.nocc, : self.nocc, self.nocc :, self.nocc :]
            ).reshape((self.ov, self.ov))
            kb = np.einsum("ibja->iajb", v).reshape((self.ov, self.ov))
            v = v.reshape((self.ov, self.ov))
            kernel = (
                (2 * v - ka - kb, 2 * v),
                (kb - ka, np.zeros_like(v)),
                (-ka, -kb),
                (
                    -ka,
                    kb,
                ),
            )

        else:
            self.log.info("RPA using provided arbitrary exchange-correlation kernel.")
            assert len(xc_kernel) == 4
            kernel = xc_kernel

        return kernel

    @property
    def mo_coeff(self):
        return self.mf.mo_coeff

    @property
    def nao(self):
        return self.mf.mol.nao

    def ao2mo(self):
        """Get the ERIs in MO basis"""

        t0 = timer()
        self.log.info("ERIs will be four centered")
        mo_coeff = self.mo_coeff
        self.eri = ao2mo.incore.full(self.mf._eri, mo_coeff, compact=False)
        self.eri = self.eri.reshape((self.nao,) * 4)
        self.log.timing("Time for AO->MO:  %s", time_string(timer() - t0))

        return self.eri
