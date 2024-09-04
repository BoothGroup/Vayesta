"""Straightforward N^6 implementation for dRPA in a basis set, based upon the standard Hermitian reformulation
used in TDHF approaches."""

import logging
from timeit import default_timer as timer

import numpy as np
import scipy.linalg
import scipy

import pyscf.ao2mo
from vayesta.core.util import dot, einsum, time_string


class ssRPA:
    """Approach based on equations expressed succinctly in the appendix of
    Furche, F. (2001). PRB, 64(19), 195120. https://doi.org/10.1103/PhysRevB.64.195120
    WARNING: Should only be used with canonical mean-field orbital coefficients in mf.mo_coeff and RHF.
    """

    def __init__(self, mf, log=None, ov_rot=None):
        self.mf = mf
        self.log = log or logging.getLogger(__name__)
        self.ov_rot = ov_rot

    @property
    def nocc(self):
        return sum(self.mf.mo_occ > 0)

    @property
    def nvir(self):
        return len(self.mf.mo_occ) - self.nocc

    @property
    def ov(self):
        return self.ova + self.ovb

    @property
    def ova(self):
        return self.nocc * self.nvir

    @property
    def ovb(self):
        return self.ova

    @property
    def mo_coeff(self):
        """MO coefficients."""
        return self.mf.mo_coeff

    @property
    def mo_coeff_occ(self):
        """Occupied MO coefficients."""
        return self.mo_coeff[:, : self.nocc]

    @property
    def mo_coeff_vir(self):
        """Virtual MO coefficients."""
        return self.mo_coeff[:, self.nocc :]

    @property
    def e_corr(self):
        try:
            return self.e_corr_ss
        except AttributeError as e:
            self.log.critical("Can only access rpa.e_corr after running rpa.kernel.")

    @property
    def e_tot(self):
        return self.mf.e_tot + self.e_corr

    def kernel(self, xc_kernel=None, alpha=1.0):
        """Solve same-spin component of dRPA response.
        At level of dRPA this is the only contribution to correlation energy; introduction of exchange will lead to
        spin-flip contributions.
        """
        t_start = timer()

        M, AmB, ApB, eps, v = self._gen_arrays(xc_kernel, alpha)

        t0 = timer()

        e, c = np.linalg.eigh(M)
        self.freqs_ss = e ** (0.5)
        assert all(e > 1e-12)

        if xc_kernel is None:
            XpY = np.einsum("n,p,pn->pn", self.freqs_ss ** (-0.5), AmB ** (0.5), c)
            XmY = np.einsum("n,p,pn->pn", self.freqs_ss ** (0.5), AmB ** (-0.5), c)
        else:
            e2, c2 = np.linalg.eigh(AmB)
            assert all(e2 > 1e-12)
            AmBrt = einsum("pn,n,qn->pq", c2, e2 ** (0.5), c2)
            AmBrtinv = einsum("pn,n,qn->pq", c2, e2 ** (-0.5), c2)
            XpY = np.einsum("n,pq,qn->pn", self.freqs_ss ** (-0.5), AmBrt, c)
            XmY = np.einsum("n,pq,qn->pn", self.freqs_ss ** (0.5), AmBrtinv, c)

        nova = self.ova
        if self.ov_rot is not None:
            nova = self.ov_rot[0].shape[0]

        self.XpY_ss = (XpY[:nova], XpY[nova:])
        self.XmY_ss = (XmY[:nova], XmY[nova:])

        self.freqs_sf = None
        self.log.timing("Time to solve RPA problem: %s", time_string(timer() - t0))

        if xc_kernel is None:
            self.e_corr_ss = (sum(self.freqs_ss) - sum(eps[0]) - sum(eps[1]) - v.trace()) / 2
        else:
            self.e_corr_ss = self.calc_energy_correction(xc_kernel=xc_kernel, version=3)

        self.log.info("Total RPA wall time:  %s", time_string(timer() - t_start))

        return self.e_corr_ss

    def calc_energy_correction(self, xc_kernel, version=3):
        if xc_kernel is None:
            raise ValueError("Without an xc-kernel the plasmon formula energy is exact.")
        t0 = timer()
        M, AmB, ApB, eps, fullv = self._gen_arrays(xc_kernel)
        ApB_xc, AmB_xc = self.get_xc_contribs(xc_kernel, self.mo_coeff_occ, self.mo_coeff_vir)
        A_xc = (ApB_xc + AmB_xc) / 2
        B_xc = (ApB_xc - AmB_xc) / 2
        full_mom0 = self.gen_moms(0, xc_kernel)[0]

        def get_eta_alpha(alpha):
            newrpa = self.__class__(self.mf, self.log)
            newrpa.kernel(xc_kernel=xc_kernel, alpha=alpha)
            mom0 = newrpa.gen_moms(0, xc_kernel=xc_kernel)[0]
            return mom0

        def run_ac_inter(func, deg=5):
            points, weights = np.polynomial.legendre.leggauss(deg)
            points += 1
            points /= 2
            weights /= 2
            return sum([w * func(p) for w, p in zip(weights, points)])

        if version == 0:
            e_plasmon = 0.5 * (np.dot(full_mom0, ApB) - (0.5 * (ApB + AmB) - A_xc)).trace()

            # Full integration of the adiabatic connection.
            def get_val_alpha(alpha):
                eta0 = get_eta_alpha(alpha)
                return (einsum("pq,qp", A_xc + B_xc, eta0) + einsum("pq,qp", A_xc - B_xc, np.linalg.inv(eta0))) / 4

            e = e_plasmon - run_ac_inter(get_val_alpha)
            e = (e, get_val_alpha)
        elif version == 1:
            e_plasmon = 0.5 * (np.dot(full_mom0, ApB) - (0.5 * (ApB + AmB) - A_xc)).trace()

            # Integration of adiabatic connection, but with approximation of the inverse eta0`
            def get_val_alpha(alpha):
                eta0 = get_eta_alpha(alpha)
                return (A_xc.trace() + einsum("pq,qp", B_xc, eta0 - np.eye(self.ov))) / 2

            e = e_plasmon - run_ac_inter(get_val_alpha)
            e = (e, get_val_alpha)
        elif version == 2:
            # Linear approximation of all quantities in adiabatic connection.
            e = 0.5 * (np.dot(full_mom0, ApB) - (0.5 * (ApB + AmB) - A_xc)).trace()
            e -= (
                einsum("pq,qp->", A_xc + B_xc, full_mom0 + np.eye(self.ov))
                + einsum("pq,qp->", A_xc - B_xc, np.linalg.inv(full_mom0) + np.eye(self.ov))
            ) / 8
        elif version == 3:
            # Linear approximation in AC and approx of inverse.
            e = 0.5 * (np.dot(full_mom0, ApB - B_xc / 2) - (0.5 * (ApB + AmB) - B_xc / 2)).trace()
        elif version == 4:

            def get_val_alpha(alpha):
                eta0 = get_eta_alpha(alpha)
                return einsum("pq,qp->", eta0 - np.eye(self.ov), fullv)

            e = 0.5 * run_ac_inter(get_val_alpha)
            e = (e, get_val_alpha)
        elif version == 5:
            e = einsum("pq,qp->", full_mom0 - np.eye(self.ov), fullv) / 4
        else:
            raise ValueError("Unknown energy approach {:s} requested.".format(version))

        self.log.timing("Time to calculate RPA energy: %s", time_string(timer() - t0))

        return e

    def _gen_eps(self):
        # Only have diagonal components in canonical basis.
        eps = np.zeros((self.nocc, self.nvir))
        eps = eps + self.mf.mo_energy[self.nocc :]
        eps = (eps.T - self.mf.mo_energy[: self.nocc]).T
        eps = eps.reshape((self.ova,))

        if self.ov_rot is not None:
            if self.ov_rot[0].shape[0] > 0:
                epsa = einsum("pn,n,qn->pq", self.ov_rot[0], eps, self.ov_rot[0])
                epsa, ca = scipy.linalg.eigh(epsa)
                self.ov_rot = (dot(ca.T, self.ov_rot[0]), self.ov_rot[1])
            if self.ov_rot[1].shape[0] > 0:
                epsb = einsum("pn,n,qn->pq", self.ov_rot[1], eps, self.ov_rot[1])
                epsb, cb = scipy.linalg.eigh(epsb)
                self.ov_rot = (self.ov_rot[0], dot(cb.T, self.ov_rot[1]))
        else:
            epsa = epsb = eps
        return epsa, epsb

    def _gen_arrays(self, xc_kernel=None, alpha=1.0):
        t0 = timer()

        epsa, epsb = self._gen_eps()

        AmB = np.concatenate([epsa, epsb])

        fullv = self.get_k()
        ApB = 2 * fullv * alpha
        if self.ov_rot is not None:
            fullrot = scipy.linalg.block_diag(self.ov_rot[0], self.ov_rot[1])
            ApB = dot(fullrot, ApB, fullrot.T)

        # At this point AmB is just epsilon so add in.
        ApB[np.diag_indices_from(ApB)] += AmB

        if xc_kernel is None:
            M = np.einsum("p,pq,q->pq", AmB ** (0.5), ApB, AmB ** (0.5))
        else:
            # Grab A and B contributions for XC kernel.
            ApB_xc, AmB_xc = self.get_xc_contribs(xc_kernel, self.mo_coeff_occ, self.mo_coeff_vir, alpha)
            ApB = ApB + ApB_xc
            AmB = np.diag(AmB) + AmB_xc
            del ApB_xc, AmB_xc
            AmBrt = scipy.linalg.sqrtm(AmB)
            M = dot(AmBrt, ApB, AmBrt)

        self.log.timing("Time to build RPA arrays: %s", time_string(timer() - t0))
        return M, AmB, ApB, (epsa, epsb), fullv

    def get_k(self):
        eris = self.ao2mo()
        # Get coulomb interaction in occupied-virtual space.
        v = eris[: self.nocc, self.nocc :, : self.nocc, self.nocc :].reshape((self.ova, self.ova))
        fullv = np.zeros((self.ov, self.ov))
        fullv[: self.ova, : self.ova] = fullv[self.ova :, self.ova :] = fullv[: self.ova, self.ova :] = fullv[
            self.ova :, : self.ova
        ] = v
        return fullv

    def get_xc_contribs(self, xc_kernel, c_o, c_v, alpha=1.0):
        if not isinstance(xc_kernel[0], np.ndarray):
            xc_kernel = [
                einsum("npq,nrs->pqrs", *xc_kernel[0]),
                einsum("npq,nrs->pqrs", xc_kernel[0][0], xc_kernel[0][1]),
                einsum("npq,nrs->pqrs", *xc_kernel[1]),
            ]
        c_o_a, c_o_b = c_o if isinstance(c_o, tuple) else (c_o, c_o)
        c_v_a, c_v_b = c_v if isinstance(c_v, tuple) else (c_v, c_v)

        ApB = np.zeros((self.ov, self.ov))
        AmB = np.zeros_like(ApB)

        V_A_aa = einsum("pqrs,pi,qa,rj,sb->iajb", xc_kernel[0], c_o_a, c_v_a, c_o_a, c_v_a).reshape(
            (self.ova, self.ova)
        )
        ApB[: self.ova, : self.ova] += V_A_aa
        AmB[: self.ova, : self.ova] += V_A_aa
        del V_A_aa
        V_B_aa = einsum("pqsr,pi,qa,rj,sb->iajb", xc_kernel[0], c_o_a, c_v_a, c_o_a, c_v_a).reshape(
            (self.ova, self.ova)
        )
        ApB[: self.ova, : self.ova] += V_B_aa
        AmB[: self.ova, : self.ova] -= V_B_aa
        del V_B_aa
        V_A_ab = einsum("pqrs,pi,qa,rj,sb->iajb", xc_kernel[1], c_o_a, c_v_a, c_o_b, c_v_b).reshape(
            (self.ova, self.ovb)
        )
        ApB[: self.ova, self.ova :] += V_A_ab
        ApB[self.ova :, : self.ova] += V_A_ab.T
        AmB[: self.ova, self.ova :] += V_A_ab
        AmB[self.ova :, : self.ova] += V_A_ab.T
        del V_A_ab
        V_B_ab = einsum("pqsr,pi,qa,rj,sb->iajb", xc_kernel[1], c_o_a, c_v_a, c_o_b, c_v_b).reshape(
            (self.ova, self.ovb)
        )
        ApB[: self.ova, self.ova :] += V_B_ab
        ApB[self.ova :, : self.ova] += V_B_ab.T
        AmB[: self.ova, self.ova :] -= V_B_ab
        AmB[self.ova :, : self.ova] -= V_B_ab.T
        del V_B_ab
        V_A_bb = einsum("pqrs,pi,qa,rj,sb->iajb", xc_kernel[2], c_o_b, c_v_b, c_o_b, c_v_b).reshape(
            (self.ovb, self.ovb)
        )
        ApB[self.ova :, self.ova :] += V_A_bb
        AmB[self.ova :, self.ova :] += V_A_bb
        del V_A_bb
        V_B_bb = einsum("pqsr,pi,qa,rj,sb->iajb", xc_kernel[2], c_o_b, c_v_b, c_o_b, c_v_b).reshape(
            (self.ovb, self.ovb)
        )
        ApB[self.ova :, self.ova :] += V_B_bb
        AmB[self.ova :, self.ova :] -= V_B_bb
        del V_B_bb

        if self.ov_rot is not None:
            if isinstance(self.ov_rot, tuple):
                fullrot = scipy.linalg.block_diag(self.ov_rot[0], self.ov_rot[1])
            else:
                fullrot = scipy.linalg.block_diag(self.ov_rot, self.ov_rot)

            ApB = dot(fullrot, ApB, fullrot.T)
            AmB = dot(fullrot, AmB, fullrot.T)

        return ApB * alpha, AmB * alpha

    def gen_moms(self, max_mom, xc_kernel=None):
        nova = self.ova
        nov = self.ov
        if self.ov_rot is not None:
            nova = self.ov_rot[0].shape[0]
            nov = nova + self.ov_rot[1].shape[0]
        res = np.zeros((max_mom + 1, nov, nov))
        for x in range(max_mom + 1):
            # Have different spin components in general; these are alpha-alpha, alpha-beta and beta-beta.
            res[x, :nova, :nova] = np.einsum("pn,n,qn->pq", self.XpY_ss[0], self.freqs_ss**x, self.XpY_ss[0])
            res[x, nova:, nova:] = np.einsum("pn,n,qn->pq", self.XpY_ss[1], self.freqs_ss**x, self.XpY_ss[1])
            res[x, :nova, nova:] = np.einsum("pn,n,qn->pq", self.XpY_ss[0], self.freqs_ss**x, self.XpY_ss[1])
            res[x, nova:, :nova] = res[x][:nova, nova:].T

        return res

    def ao2mo(self, mo_coeff=None, compact=False):
        """Get the ERIs in MO basis"""
        mo_coeff = self.mo_coeff if mo_coeff is None else mo_coeff

        t0 = timer()
        if hasattr(self.mf, "with_df") and self.mf.with_df is not None:
            eris = self.mf.with_df.ao2mo(mo_coeff, compact=compact)
        elif self.mf._eri is not None:
            eris = pyscf.ao2mo.kernel(self.mf._eri, mo_coeff, compact=compact)
        else:
            eris = self.mf.mol.ao2mo(mo_coeff, compact=compact)
        if not compact:
            if isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 2:
                shape = 4 * [mo_coeff.shape[-1]]
            else:
                shape = [mo.shape[-1] for mo in mo_coeff]
            eris = eris.reshape(shape)
        self.log.timing("Time for AO->MO of ERIs:  %s", time_string(timer() - t0))
        return eris


ssRRPA = ssRPA
