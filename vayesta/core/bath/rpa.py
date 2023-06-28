from vayesta.core.bath.bno import BNO_Threshold, BNO_Bath

from vayesta.core.util import AbstractMethodError, brange, dot, einsum, fix_orbital_sign, hstack, time_string, timer
from vayesta.core.types import Cluster

from vayesta.rpa.rirpa import ssRIdRRPA

from vayesta.core.eris import get_cderi
from vayesta.core import spinalg
from vayesta.core.bath import helper


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

        if self.occtype == "occupied":
            proj = dot(self.dmet_bath.c_cluster_vir.T, self.base.get_ovlp(), self.fragment.c_frag,
                       self.fragment.c_frag.T, self.base.get_ovlp(), self.dmet_bath.c_cluster_vir)

            rot_vir = dot(self.dmet_bath.c_cluster_vir.T, self.base.get_ovlp(), self.base.mo_coeff_vir)
            rot_occ = np.eye(self.base.nocc)
        else:
            proj = dot(self.dmet_bath.c_cluster_occ.T, self.base.get_ovlp(), self.fragment.c_frag, self.fragment.c_frag.T,
                             self.base.get_ovlp(), self.dmet_bath.c_cluster_occ)
            rot_occ = dot(self.dmet_bath.c_cluster_occ.T, self.base.get_ovlp(), self.base.mo_coeff_occ)
            rot_vir = np.eye(self.base.nvir)

        loc_excit_shape = (rot_occ.shape[0], rot_vir.shape[0])

        # Get target rotation in particle-hole excitation space.
        # This is of size O(N), so this whole procedure scales as O(N^4)

        target_rot = einsum("ij,ab->iajb", rot_occ, rot_vir)
        target_rot = target_rot.reshape(np.product(target_rot.shape[:2]), np.product(target_rot.shape[2:]))

        print(proj.shape, rot_vir.shape, rot_occ.shape, loc_excit_shape, target_rot.shape)

        t0 = timer()
        myrpa = ssRIdRRPA(self.base.mf, lov=cderis)
        # This initially calculates the spin-summed zeroth moment, then deducts the spin-dependent component and
        # accounts for factor of two from different spin channels.
        m0 = (myrpa.kernel_moms(0, target_rot=target_rot, return_spatial=True)[0][0] - target_rot) / 2.0
        self.fragment._rpa_bath_intermed = m0
        print("m0 shape (generated):", m0.shape)

        # Get eps with occupied index in DMET cluster.
        eps = dot(target_rot, myrpa.eps.reshape(-1)).reshape((rot_occ.shape[0], rot_vir.shape[0]))
        # First calculate contributions to occupied and virtual matrices.
        if self.occtype == "occupied":
            econtrib = np.diag(einsum("ab,iaib->i", proj,
                              dot(m0 * myrpa.eps[None], target_rot.T).reshape(loc_excit_shape+loc_excit_shape)))
        else:
            econtrib = np.diag(einsum("ij,iaja->a", proj,
                              dot(m0 * myrpa.eps[None], target_rot.T).reshape(loc_excit_shape+loc_excit_shape)))

        cderi_loc = dot(cderis[0].reshape(-1, myrpa.ov), target_rot.T).reshape((cderis[0].shape[0],)+loc_excit_shape)

        if cderis[1] is not None:
            cderi_neg_loc = dot(cderis[1].reshape(-1, myrpa.ov), target_rot.T).reshape(
                (cderis[1].shape[0],) + loc_excit_shape)

        #m0 = dot(m0, target_rot.T).reshape(loc_excit_shape + loc_excit_shape)
        #mycderis = (cderi_loc, cderi_neg_loc)
        m0 = m0.reshape(loc_excit_shape + (myrpa.nocc, myrpa.nvir))
        mycderis = cderis


        if self.occtype == "occupied":

            # Since loc excit space only scales as O(N), all steps are limited to N^4 scaling.
            m02 = einsum("jbia,nia->jbni", m0, mycderis[0])
            econtrib += 4 * einsum("njc,cb,jbni->ji", cderi_loc, proj, m02)
            if cderis[1] is not None:
                m02 = einsum("jbia,nia->jbni", m0, mycderis[1])
                econtrib -= 4 * einsum("njc,cb,jbni->ji", cderi_neg_loc, proj, m02)

        else:
            # Since loc excit space only scales as O(N), all steps are limited to N^4 scaling.
            m02 = einsum("jbia,nia->jbna", m0, mycderis[0])

            econtrib += 4 * einsum("nkb,kj,jbna->ba", cderi_loc, proj, m02)
            if cderis[1] is not None:
                m02 = einsum("jbia,nia->jbna", m0, mycderis[1])
                econtrib -= 4 * einsum("nkb,kj,jbna->ba", cderi_neg_loc, proj, m02)

        # Finally add in pure coulombic contribution.
        if self.occtype == "occupied":
            econtrib += einsum("nia,njb,ab->ij", cderi_loc, cderi_loc, proj)
            if cderis[1] is not None:
                econtrib -= einsum("nia,njb,ab->ij", cderi_neg_loc, cderi_neg_loc, proj)

        else:
            econtrib += einsum("nia,njb,ij->ab", cderi_loc, cderi_loc, proj)
            if cderis[1] is not None:
                econtrib -= einsum("nia,njb,ij->ab", cderi_neg_loc, cderi_neg_loc, proj)

        t_eval = timer()-t0

        self.log.info("Energy contrib asymmetry: %e", np.max(np.abs(econtrib - econtrib.T)))

        # econtrib is asymmetric; we can symmetrise as separation is arbitrary.
        econtrib = (econtrib + econtrib.T) / 2.0

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

        an_bno = abs(n_bno)

        c_bno = c_bno[:, np.argsort(an_bno)[::-1]]
        an_bno = np.array(sorted(an_bno)[::-1])


        return c_bno, an_bno, 0.0


    def log_histogram(self, n_bno):
        if len(n_bno) == 0:
            return
        self.log.info("%s BNO histogram:", self.occtype.capitalize())

        min = int(np.floor(np.log10(n_bno.min())))
        max = int(np.ceil(np.log10(n_bno.max())))

        print("limits:", min, max)

        if max - min > 8:
            max = min + 8

        bins = np.hstack([-np.inf, np.logspace(max, min, 1 + max-min)[::-1], np.inf])
        print("bins:", bins)
        labels = '    ' + ''.join('{:{w}}'.format('E-%d' % d, w=5) for d in range(-max, -min, 1))
        self.log.info(helper.make_histogram(n_bno, bins=bins, labels=labels))

