from vayesta.core.bath.bno import BNO_Threshold, BNO_Bath

from vayesta.core.util import AbstractMethodError, brange, dot, einsum, fix_orbital_sign, hstack, time_string, timer
from vayesta.core.types import Cluster

from vayesta.rpa.rirpa import ssRIdRRPA

from vayesta.core.eris import get_cderi
from vayesta.core import spinalg


import numpy as np

class RPA_BNO_Bath(BNO_Bath):
    def __init__(self, *args, project_dmet_order=0, project_dmet_mode='full', project_dmet=None, **kwargs):
        self.project_dmet_order = project_dmet_order
        self.project_dmet_mode = project_dmet_mode
        super().__init__(*args, **kwargs)

    def make_bno_coeff(self, cderis=None):
        """Construct MP2 bath natural orbital coefficients and occupation numbers.

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

        print("Entering RPA_BNO_Bath.make_bno_coeff")
        t_init = timer()

        if cderis is None:
            cderis = get_cderi(self.base, (self.base.mo_coeff_occ, self.base.mo_coeff_vir), compact=False)

        c_active_occ = self.dmet_bath.c_cluster_occ

        fock = self.base.get_fock_for_bath()

        target_occ = dot(c_active_occ.T, self.base.get_ovlp(), self.base.mo_coeff_occ)
        nocc_dmet = target_occ.shape[0]
        nvir = self.base.nvir

        print("c_active_occ.shape:", c_active_occ.shape)

        # Get target rotation in particle-hole excitation space.
        # This is of size O(N), so this whole procedure scales as O(N^4)
        target = einsum("ij,ab->iajb", target_occ, np.eye(nvir))
        target = target.reshape((target_occ.shape[0] * nvir, self.base.nocc * nvir))
        t0 = timer()
        myrpa = ssRIdRRPA(self.base.mf, lov=cderis)

        # Hacky workaround for restrictive bath construction interface only allowing totally decoupled occupied/virtual
        # bath constructions...
        if hasattr(self.fragment, "_rpa_bath_intermed"):
            m0 = self.fragment._rpa_bath_intermed
            print("m0 shape (read):", m0.shape)
        else:
            # This initially calculates the spin-summed zeroth moment, then deducts the spin-dependent component and
            # accounts for factor of two.
            m0 = (myrpa.kernel_moms(0, target_rot=target, return_spatial=True)[0][0] - target) / 2.0
            self.fragment._rpa_bath_intermed = m0
            print("m0 shape (generated):", m0.shape)
        # Get eps with occupied index in DMET cluster.
        eps = einsum("ia,ji->ja",myrpa.eps.reshape((myrpa.nocc, myrpa.nvir)), target_occ)
        # First calculate contributions to occupied and virtual matrices.
        if self.occtype == "occupied":
            econtrib = dot(target_occ.T,
                               einsum("ia,iaja->ij", eps,
                                      m0.reshape((target_occ.shape[0], myrpa.nvir, myrpa.nocc, myrpa.nvir))))
        else:
            econtrib = einsum("ia,iajb,ij->ab", eps,
                                  m0.reshape((target_occ.shape[0], myrpa.nvir, myrpa.nocc, myrpa.nvir)), target_occ)

        # Generate cderi in occupied-cluster virtual-full space.
        vloc = einsum("nia,ji->nja", cderis[0], target_occ).reshape((cderis[0].shape[0], -1))
        m02 = dot(vloc, m0).reshape(vloc.shape[0], myrpa.nocc, myrpa.nvir)

        m02_neg = None
        if cderis[1] is not None:
            vloc_neg = einsum("nia,ji->nja", cderis[1], target_occ).reshape((cderis[1].shape[0], -1))
            m02_neg = dot(vloc_neg, m0).reshape(vloc_neg.shape[0], myrpa.nocc, myrpa.nvir)

        if self.occtype == "occupied":
            econtrib += 4 * einsum("nia,nja->ij", cderis[0], m02)
            if m02_neg is not None:
                econtrib -= 4 * einsum("nia,nja->ij", cderis[1], m02_neg)
        else:
            econtrib += 4 * einsum("nia,nib->ab", cderis[0], m02)
            if m02_neg is not None:
                econtrib -= 4 * einsum("nia,nib->ab", cderis[1], m02_neg)

        # Finally add in pure coulombic contribution.
        temp = dot(vloc.T, vloc).reshape((nocc_dmet, nvir, nocc_dmet, nvir))
        if self.occtype == "occupied":
            econtrib += dot(target_occ.T, einsum("iaja->ij", temp), target_occ)
        else:
            econtrib += einsum("iaib->ab", temp)

        t_eval = timer()-t0

        # econtrib is asymmetric; we can symmetrise.
        econtrib = (econtrib + econtrib.T) / 2.0


        print(myrpa.nocc, myrpa.nvir, target_occ.shape)
        print(econtrib.shape)

        # --- Diagonalize environment-environment block
        if self.occtype == 'occupied':
            econtrib = self._rotate_dm(econtrib, dot(self.dmet_bath.c_env_occ.T, self.base.get_ovlp(), self.base.mo_coeff_occ))
        elif self.occtype == 'virtual':
            econtrib = self._rotate_dm(econtrib, dot(self.dmet_bath.c_env_vir.T, self.base.get_ovlp(), self.base.mo_coeff_vir))
        t0 = timer()
        r_bno, n_bno = self._diagonalize_dm(econtrib)
        t_diag = timer()-t0
        c_bno = spinalg.dot(self.c_env, r_bno)
        c_bno = fix_orbital_sign(c_bno)[0]

        self.log.timing("Time RPA bath:  evaluation= %s  diagonal.= %s  total= %s",
                *map(time_string, (t_eval, t_diag, (timer()-t_init))))

        print(n_bno)

        return c_bno, abs(n_bno), 0.0




