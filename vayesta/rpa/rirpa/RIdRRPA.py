import numpy as np

from vayesta.rpa.rirpa.RIRPA import ssRIRRPA
from vayesta.rpa.rirpa.momzero_NI_direct_restricted import MomzeroDeductNone_dRHF, MomzeroDeductD_dRHF, MomzeroDeductHigherOrder_dRHF
from vayesta.rpa.rirpa.momzero_NI import MomzeroOffsetCalcGaussLag
from vayesta.core.util import dot, einsum, time_string, timer, with_doc

from pyscf import lib


class ssRIdRRPA(ssRIRRPA):
    """Class for computing direct RPA correlated quantites with a restricted reference state.
    """

    @with_doc(ssRIRRPA.kernel_moms)
    def kernel_moms(self, max_moment, target_rot=None, **kwargs):
        t_start = timer()

        tr_rot = None

        if target_rot is None:
            self.log.warning(
                "Warning; generating full moment rather than local component. Will scale as O(N^5)."
            )
            target_rot = np.eye(self.ov)
        else:
            target_rot = self.check_target_rot(target_rot)
            if self.compress > -1:
                target_rot, tr_rot = self.compress_target_rot(target_rot)

        ri_decomps = self.get_compressed_MP()
        ri_mp, ri_apb, ri_amb = ri_decomps
        # First need to calculate zeroth moment.
        moments = np.zeros((max_moment + 1,) + target_rot.shape)
        moments[0], err0 = self._kernel_mom0(
            target_rot, ri_decomps=ri_decomps, **kwargs
        )

        t_start_higher = timer()
        if max_moment > 0:
            # Grab mean.
            eps = self.eps
            moments[1] = target_rot * eps[None]
            if max_moment > 1:
                for i in range(2, max_moment + 1):
                    moments[i] = moments[i - 2] * (eps**2)[None] + 2 * dot(moments[i - 2], ri_mp[1].T, ri_mp[0])
        self.record_memory()
        if max_moment > 0:
            self.log.info(
                "RIRPA Higher Moments wall time:  %s",
                time_string(timer() - t_start_higher),
            )
            self.log.info(
                "Overall RIRPA Moments wall time:  %s", time_string(timer() - t_start)
            )
        if tr_rot is not None:
            moments = np.tensordot(tr_rot, moments, ((1,), (0,)))
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
    ):
        t_start = timer()
        if analytic_lower_bound or adaptive_quad:
            raise NotImplementedError("Only main functionality is implemented for dRPA specific code.")

        if target_rot is None:
            self.log.warning(
                "Warning; generating full moment rather than local component. Will scale as O(N^5)."
            )
            target_rot = np.eye(self.ov)
        else:
            target_rot = self.check_target_rot(target_rot)

        if ri_decomps is None:
            ri_mp, ri_apb, ri_amb = self.get_compressed_MP(alpha)
        else:
            ri_mp, ri_apb, ri_amb = ri_decomps
        self.record_memory()

        offset_niworker = None
        inputs = (self.eps, ri_mp[0], ri_mp[1], target_rot, npoints, self.log)
        if integral_deduct == "D":
            # Evaluate (MP)^{1/2} - D,
            niworker = MomzeroDeductD_dRHF(*inputs)
            integral_offset = target_rot * self.eps[None]
        elif integral_deduct is None:
            # Explicitly evaluate (MP)^{1/2}, with no offsets.
            niworker = MomzeroDeductNone_dRHF(*inputs)
            integral_offset = np.zeros_like(target_rot)
        elif integral_deduct == "HO":
            niworker = MomzeroDeductHigherOrder_dRHF(*inputs)
            offset_niworker = MomzeroOffsetCalcGaussLag(*inputs)
            estval, offset_err = offset_niworker.kernel()
            integral_offset = target_rot * self.eps[None] + estval
        else:
            raise ValueError("Unknown integral offset specification.`")

        if return_niworker:
            return niworker, offset_niworker
        self.record_memory()

        integral, upper_bound = niworker.kernel(a=ainit, opt_quad=opt_quad)

        self.record_memory()

        # Free memory.
        del ri_mp, inputs, niworker
        self.record_memory()
        print("Norm integral offset", np.linalg.norm(integral_offset))
        print("Norm integral", np.linalg.norm(integral))
        integral += integral_offset
        del integral_offset

        # Construct A+B inverse.
        eps = self.eps
        epsinv = eps ** (-1)
        # Factor of two from different spin channels.
        u = 2 * dot(ri_apb[0] * epsinv[None], ri_apb[1].T)
        u = np.linalg.inv(np.eye(u.shape[0]) + u)
        print("U", np.linalg.norm(u), sum(u.ravel()))
        self.record_memory()
        mom0 = integral / self.eps[None]
        # Factor of two from sum over spin channels.
        mom0 -= 2 * dot(
            dot(
                dot(
                    integral, (ri_apb[0] * epsinv[None]).T
                    ),
                u
            ),
            ri_apb[1] * epsinv[None]
        )

        self.log.info(
            "RIRPA Zeroth Moment wall time:  %s", time_string(timer() - t_start)
        )

        return mom0, (None, None)

    def get_compressed_MP(self, alpha=1.0):
        lov, lov_neg = self.get_cderi()

        lov = lov.reshape(lov.shape[0], -1)
        ri_apb = [lov, lov]

        if lov_neg is not None:
            lov_neg.reshape(lov_neg.shape[0], -1)

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
        if target_rot is None:
            self.log.warning(
                "Warning; generating full moment rather than local component. Will scale as O(N^5)."
            )
            target_rot = np.eye(self.ov)
        elif target_rot.shape[1] == self.ov_tot:
            # Provided rotation in spinorbital space.
            target_rot = target_rot[:, :self.ov] + target_rot[:, self.ov:]

        return target_rot

    def compress_target_rot(self, target_rot, tol=1e-10):
        inner_prod = dot(target_rot.T, target_rot)
        e, c = np.linalg.eigh(inner_prod)
        want = e > tol
        rot = c[:, want]
        if rot.shape[1] == target_rot.shape[1]:
            return target_rot, None
        return dot(target_rot, rot), rot
