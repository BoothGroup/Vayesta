import numpy as np

from vayesta.rpa.rirpa.RIRPA import ssRIRRPA
from vayesta.rpa.rirpa.momzero_NI import MomzeroOffsetCalcGaussLag, MomzeroDeductHigherOrder_dRHF
from vayesta.rpa.rirpa.energy_NI import NITrRootMP_dRHF
from vayesta.core.util import dot, time_string, timer, with_doc

from pyscf import lib


class ssRIdRRPA(ssRIRRPA):
    """Class for computing direct RPA correlated quantites with a restricted reference state.
    """

    @with_doc(ssRIRRPA.kernel_moms)
    def kernel_moms(self, max_moment, target_rot=None, return_spatial=False, **kwargs):
        t_start = timer()
        self.log.debug("Running specialised dRPA RHF code.")

        if target_rot is None:
            self.log.warning(
                "Warning; generating full moment rather than local component. Will scale as O(N^5)."
            )
            if return_spatial:
                target_rot = np.eye(self.ov)
            else:
                target_rot = np.eye(self.ov_tot)

        ri_decomps = self.get_compressed_MP()
        ri_mp, ri_apb, ri_amb = ri_decomps
        # First need to calculate zeroth moment. This only generates the spin-independent contribution in a single
        # spin channel; the spin-dependent contribution is just the identity rotated to our target rotation.
        mom0, err0 = self._kernel_mom0(
            target_rot, ri_decomps=ri_decomps, return_spatial=return_spatial, **kwargs
        )

        moments = np.zeros((max_moment + 1,) + mom0.shape, dtype=mom0.dtype)

        moments[0] = mom0

        t_start_higher = timer()

        if max_moment == 0:
            return moments, err0

        eps = self.eps

        if return_spatial:
            # Must have spatial target rotation.
            def gen_new_moment(prev_mom):
                return prev_mom * (eps**2)[None] + 2 * dot(prev_mom, ri_mp[1].T, ri_mp[0])

            moments[1] = target_rot * eps[None]

        else:
            def gen_new_moment(prev_mom):
                prev_aa, prev_bb = prev_mom[:, :self.ov], prev_mom[:, self.ov:]
                spat_vv = dot(prev_aa + prev_bb, ri_mp[1].T, ri_mp[0])
                new_aa = spat_vv + prev_aa * (eps**2)[None]
                new_bb = spat_vv + prev_bb * (eps**2)[None]
                return np.concatenate((new_aa, new_bb), axis=1)

            if target_rot.shape[1] == self.ov:
                moments[1, :, :self.ov] = target_rot * eps[None]
            else:
                moments[1] = target_rot * self.D[None]

        if max_moment > 1:
            for i in range(2, max_moment + 1):
                moments[i] = gen_new_moment(moments[i - 2])
        self.record_memory()
        if max_moment > 0:
            self.log.info(
                "RIRPA Higher Moments wall time:  %s",
                time_string(timer() - t_start_higher),
            )
            self.log.info(
                "Overall RIRPA Moments wall time:  %s", time_string(timer() - t_start)
            )
        return moments, err0

    @with_doc(ssRIRRPA._kernel_mom0)
    def _kernel_mom0(
        self,
        target_rot=None,
        npoints=48,
        ainit=10,
        integral_deduct="HO",
        opt_quad=True,
        adaptive_quad=False,
        alpha=1.0,
        ri_decomps=None,
        return_niworker=False,
        analytic_lower_bound=False,
        return_spatial=False
    ):
        t_start = timer()
        if analytic_lower_bound or adaptive_quad or integral_deduct != "HO":
            raise NotImplementedError("Only core functionality is implemented in dRPA specific code.")
        # If we have a rotation in the spinorbital basis we need to stack the different spin channels as additional
        # spatial rotations and then sum at the end.
        target_rot, stack_spin = self.check_target_rot(target_rot)
        trrot = None
        # We can then compress the spatial rotation.
        if stack_spin:
            if return_spatial:
                raise ValueError("Requested spatially integrated calculation, but target_rot is spin-dependent.")
            target_rot, trrot = self.compress_target_rot(target_rot)

        if ri_decomps is None:
            ri_mp, ri_apb, ri_amb = self.get_compressed_MP(alpha)
        else:
            ri_mp, ri_apb, ri_amb = ri_decomps
        self.record_memory()

        offset_niworker = None
        inputs = (self.eps, ri_mp[0], ri_mp[1], target_rot, npoints, self.log)

        # We are computing our values using spatial quantities, but relating our calculations to the spinorbital
        # resolved basis. As such, we need to be careful to ensure we know which terms are spin-diagonal and which are
        # spin-invariant.

        niworker = MomzeroDeductHigherOrder_dRHF(*inputs)
        offset_niworker = MomzeroOffsetCalcGaussLag(*inputs)

        if return_niworker:
            return niworker, offset_niworker
        self.record_memory()

        integral, upper_bound = niworker.kernel(a=ainit, opt_quad=opt_quad)  # This contribution is spin-invariant.
        integral += offset_niworker.kernel()[0]  # As is this one.

        self.record_memory()

        # Free memory.
        del ri_mp, inputs, niworker
        self.record_memory()

        # Construct A+B inverse.
        eps = self.eps
        epsinv = eps ** (-1)
        # Factor of two from different spin channels.
        u = 2 * dot(ri_apb[0] * epsinv[None], ri_apb[1].T)
        u = np.linalg.inv(np.eye(u.shape[0]) + u)
        self.record_memory()

        # First, compute contribution which is spin-dependent (diagonal in both the integrated value, and (A+B)^-1).
        mom0_spinindependent = integral * epsinv[None]

        # Factor of two from sum over spin channels.
        mom0_spinindependent -= dot(
            dot(
                dot(
                    target_rot * self.eps[None] + 2 * integral, (ri_apb[1] * epsinv[None]).T
                    ),
                u
            ),
            ri_apb[0] * epsinv[None]
        )


        self.log.info(
            "RIRPA Zeroth Moment wall time:  %s", time_string(timer() - t_start)
        )

        if trrot is not None:
            target_rot = dot(trrot, target_rot)
            mom0_spinindependent = dot(trrot, mom0_spinindependent)

        if return_spatial:
            mom0 = target_rot + 2 * mom0_spinindependent
        else:
            # Want to return quantities in spin-orbital basis.
            n = target_rot.shape[0]
            if stack_spin:
                # Half of target rot are actually other spin.
                 n = n// 2
            mom0 = np.zeros((n, self.ov_tot))

            if stack_spin:
                mom0[:, :self.ov] = target_rot[:n] + mom0_spinindependent[:n] + mom0_spinindependent[n:]  # Has aa and ab interactions.
                mom0[:, self.ov:] = target_rot[n:] + mom0_spinindependent[n:] + mom0_spinindependent[:n]  # Has bb and ba interactions.
            else:
                mom0[:, :self.ov] = target_rot + mom0_spinindependent
                mom0[:, self.ov:] = mom0_spinindependent

        return mom0, (None, None)

    def kernel_trMPrt(self, npoints=48, ainit=10):
        """Evaluate Tr((MP)^(1/2))."""
        ri_mp, ri_apb, ri_amb = self.get_compressed_MP()
        inputs = (self.eps, ri_mp[0], ri_mp[1], npoints, self.log)

        niworker = NITrRootMP_dRHF(*inputs)
        integral, err = niworker.kernel(a=ainit, opt_quad=True)
        # Compute offset; possible analytically in N^3 for diagonal.
        offset = 2 * sum(self.eps) + np.tensordot(ri_mp[0] * self.eps ** (-1), ri_mp[1], ((0, 1), (0, 1)))
        return integral[0] + offset, err

    def kernel_energy(self, npoints=48, ainit=10, correction="linear"):

        t_start = timer()
        e1, err = self.kernel_trMPrt(npoints, ainit)
        e2 = 0.0
        cderi, cderi_neg = self.get_cderi()
        cderi = cderi.reshape((-1, self.ov))
        if cderi_neg is not None:
            cderi_neg = cderi_neg.reshape((-1, self.ov))
        # Note that eri contribution to A and B is equal, so can get trace over one by dividing by two
        e3 = 2 * (sum(self.eps) + np.tensordot(cderi, cderi, ((0, 1), (0, 1))))
        if cderi_neg is not None:
             e3 -= np.tensordot(cderi_neg, cderi_neg, ((0, 1), (0, 1)))
        err /= 2
        self.e_corr_ss = 0.5 * (e1 + e2 - e3)
        self.log.info(
            "Total RIRPA Energy Calculation wall time:  %s",
            time_string(timer() - t_start),
        )
        return self.e_corr_ss, err

    def get_compressed_MP(self, alpha=1.0):
        lov, lov_neg = self.get_cderi()

        lov = lov.reshape(lov.shape[0], self.ov)
        ri_apb = [lov, lov]

        if lov_neg is not None:
            lov_neg = lov_neg.reshape(lov_neg.shape[0], self.ov)

            ri_apb = [np.concatenate([lov, lov_neg], axis=0),
                      np.concatenate([lov, -lov_neg], axis=0)]

        if self.compress > 3:
            ri_apb = self.compress_low_rank(*ri_apb, name="A+B")

        ri_apb = [x * np.sqrt(2) * (alpha ** (0.5)) for x in ri_apb]
        ri_amb = [np.zeros((0, lov.shape[1]))] * 2

        ri_mp = [ri_apb[0] * self.eps[None], ri_apb[1]]
        if self.compress > 0:
            ri_mp = self.compress_low_rank(*ri_mp, name="(A-B)(A+B)")
        return ri_mp, ri_apb, ri_amb

    def check_target_rot(self, target_rot):
        stack_spins = False
        if target_rot is None:
            self.log.warning(
                "Warning; generating full moment rather than local component. Will scale as O(N^5)."
            )
            target_rot = np.eye(self.ov)
        elif target_rot.shape[1] == self.ov_tot:
            # Provided rotation is in spinorbital space. We want to convert to spatial, but record that we've done this
            # so we can convert back later.
            target_rot = np.concatenate([target_rot[:, :self.ov], target_rot[:, self.ov:]], axis=0)
            stack_spins = True
        return target_rot, stack_spins

    def compress_target_rot(self, target_rot, tol=1e-10):
        inner_prod = dot(target_rot, target_rot.T)
        e, c = np.linalg.eigh(inner_prod)
        want = e > tol
        rot = c[:, want]
        if rot.shape[1] == target_rot.shape[1]:
            return target_rot, None
        return dot(rot.T, target_rot), rot
