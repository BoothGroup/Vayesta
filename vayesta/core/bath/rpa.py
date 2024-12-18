from vayesta.core.bath.bno import BNO_Threshold, BNO_Bath

from vayesta.core.util import AbstractMethodError, brange, dot, einsum, fix_orbital_sign, hstack, time_string, timer
from vayesta.core.types import Cluster

from vayesta.rpa.rirpa import ssRIdRRPA

from vayesta.core.eris import get_cderi
from vayesta.core import spinalg
from vayesta.core.bath import helper


import numpy as np


class RPA_BNO_Bath(BNO_Bath):
    def __init__(self, *args, project_dmet_order=0, project_dmet_mode="full", project_dmet=None, **kwargs):
        self.project_dmet_order = project_dmet_order
        self.project_dmet_mode = project_dmet_mode
        super().__init__(*args, **kwargs)

    def make_bno_coeff(self, cderis=None):
        """Construct RPA bath natural orbital coefficients and occupation numbers.

        This routine works for both for spin-restricted and unrestricted.

        Parameters
        ----------
        cderis: cderis in the particle-hole space.

        Returns
        -------
        c_bno: (n(AO), n(BNO)) array
            Bath natural orbital coefficients.
        n_bno: (n(BNO)) array
            Bath natural orbital occupation numbers.
        """

        t_init = timer()

        if cderis is None:
            cderis = get_cderi(self.base, (self.base.mo_coeff_occ, self.base.mo_coeff_vir), compact=False)

        if self.occtype == "occupied":
            proj = dot(
                self.dmet_bath.c_cluster_vir.T,
                self.base.get_ovlp(),
                self.fragment.c_frag,
                self.fragment.c_frag.T,
                self.base.get_ovlp(),
                self.dmet_bath.c_cluster_vir,
            )

            rot_vir = dot(self.dmet_bath.c_cluster_vir.T, self.base.get_ovlp(), self.base.mo_coeff_vir)
            rot_occ = np.eye(self.base.nocc)
        else:
            proj = dot(
                self.dmet_bath.c_cluster_occ.T,
                self.base.get_ovlp(),
                self.fragment.c_frag,
                self.fragment.c_frag.T,
                self.base.get_ovlp(),
                self.dmet_bath.c_cluster_occ,
            )
            rot_occ = dot(self.dmet_bath.c_cluster_occ.T, self.base.get_ovlp(), self.base.mo_coeff_occ)
            rot_vir = np.eye(self.base.nvir)

        loc_excit_shape = (rot_occ.shape[0], rot_vir.shape[0])

        # Get target rotation in particle-hole excitation space.
        # This is of size O(N), so this whole procedure scales as O(N^4)

        target_rot = einsum("ij,ab->iajb", rot_occ, rot_vir)
        target_rot = target_rot.reshape(np.prod(target_rot.shape[:2]), np.prod(target_rot.shape[2:]))

        t0 = timer()
        myrpa = ssRIdRRPA(self.base.mf, lov=cderis)
        # This initially calculates the spin-summed zeroth moment, then deducts the spin-dependent component and
        # accounts for factor of two from different spin channels.
        m0 = (myrpa.kernel_moms(0, target_rot=target_rot, return_spatial=True)[0][0] - target_rot) / 2.0
        m0 = -dot(m0, target_rot.T).reshape(loc_excit_shape + loc_excit_shape)
        if self.occtype == "occupied":
            corr_dm = einsum("iajb,ab->ij", m0, proj)
        else:
            corr_dm = einsum("iajb,ij->ab", m0, proj)
        t_eval = timer() - t0

        corr_dm = (corr_dm + corr_dm.T) / 2

        # --- Diagonalize environment-environment block
        if self.occtype == "occupied":
            corr_dm = self._rotate_dm(
                corr_dm, dot(self.dmet_bath.c_env_occ.T, self.base.get_ovlp(), self.base.mo_coeff_occ)
            )
        elif self.occtype == "virtual":
            corr_dm = self._rotate_dm(
                corr_dm, dot(self.dmet_bath.c_env_vir.T, self.base.get_ovlp(), self.base.mo_coeff_vir)
            )
        t0 = timer()
        r_bno, n_bno = self._diagonalize_dm(corr_dm)
        t_diag = timer() - t0
        c_bno = spinalg.dot(self.c_env, r_bno)
        c_bno = fix_orbital_sign(c_bno)[0]

        self.log.timing(
            "Time RPA bath:  evaluation= %s  diagonal.= %s  total= %s",
            *map(time_string, (t_eval, t_diag, (timer() - t_init))),
        )

        if min(n_bno) < 0.0:
            self.log.critical("Negative bath occupation number encountered: %s", n_bno)

        return c_bno, n_bno, 0.0
