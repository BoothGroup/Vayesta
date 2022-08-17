import numpy as np

import pyscf
import pyscf.cc

import vayesta.core.ao2mo
from vayesta.core.util import *
from vayesta.core.qemb import UFragment as BaseFragment
from .fragment import Fragment as RFragment


class Fragment(RFragment, BaseFragment):

    def set_cas(self, *args, **kwargs):
        raise NotImplementedError()

    def get_fragment_energy(self, c1, c2, eris=None, fock=None, axis1='fragment', c2ba_order='ba'):
        """Calculate fragment correlation energy contribution from projected C1, C2.

        Parameters
        ----------
        c1: (n(occ-CO), n(vir-CO)) array
            Fragment projected C1-amplitudes.
        c2: (n(occ-CO), n(occ-CO), n(vir-CO), n(vir-CO)) array
            Fragment projected C2-amplitudes.
        eris: array or PySCF _ChemistERIs object
            Electron repulsion integrals as returned by ccsd.ao2mo().
        fock: (n(AO), n(AO)) array, optional
            Fock matrix in AO representation. If None, self.base.get_fock_for_energy()
            is used. Default: None.

        Returns
        -------
        e_singles: float
            Fragment correlation energy contribution from single excitations.
        e_doubles: float
            Fragment correlation energy contribution from double excitations.
        e_corr: float
            Total fragment correlation energy contribution.
        """
        nocc = (c2[0].shape[1], c2[-1].shape[1])
        nvir = (c2[0].shape[2], c2[-1].shape[2])
        self.log.debugv("nocc= %d, %d nvir= %d, %d", *nocc, *nvir)
        oa, ob = np.s_[:nocc[0]], np.s_[:nocc[1]]
        va, vb = np.s_[nocc[0]:], np.s_[nocc[1]:]
        if axis1 == 'fragment':
            pxa, pxb = self.get_overlap('frag|cluster-occ')

        # --- Singles energy (zero for HF-reference)
        if c1 is not None:
            #if hasattr(eris, 'fock'):
            #    fa = eris.fock[0][oa,va]
            #    fb = eris.fock[1][ob,vb]
            #else:
            #    fock = self.base.get_fock()
            #    fa = dot(self.c_active_occ[0].T, fock[0], self.c_active_vir[0])
            #    fb = dot(self.c_active_occ[1].T, fock[1], self.c_active_vir[1])
            if fock is None:
                fock = self.base.get_fock_for_energy()
            fova = dot(self.cluster.c_active_occ[0].T, fock[0], self.cluster.c_active_vir[0])
            fovb = dot(self.cluster.c_active_occ[1].T, fock[1], self.cluster.c_active_vir[1])
            assert (len(c1) == 2)
            ca, cb = c1
            if axis1 == 'fragment':
                e_singles = (einsum('ia,xi,xa->', fova, pxa, ca)
                           + einsum('ia,xi,xa->', fovb, pxb, cb))
            else:
                e_singles = np.sum(fova*ca) + np.sum(fovb*cb)
        else:
            e_singles = 0
        # Doubles energy
        # TODO: loop to reduce memory?
        if eris is None:
            eris = self._eris
        if hasattr(eris, 'ovov'):
            gaa = eris.ovov
            #gaa = eris.ovov - eris.ovov.transpose(0,3,2,1)
            gab = eris.ovOV
            gbb = eris.OVOV
            #gbb = eris.OVOV - eris.OVOV.transpose(0,3,2,1)
        elif eris[0].shape == (nocc[0], nvir[0], nocc[0], nvir[0]):
            gaa, gab, gbb = eris
        else:
            assert (len(eris) == 3)
            gaa = eris[0][oa,va,oa,va]
            gab = eris[1][oa,va,ob,vb]
            gbb = eris[2][ob,vb,ob,vb]

        #caa = caa + einsum('ia,jb->ijab', ca, ca) - einsum('ib,ja->ijab', ca, ca)
        #cbb = cbb + einsum('ia,jb->ijab', cb, cb) - einsum('ib,ja->ijab', cb, cb)
        #e_d = (einsum('ijab,iajb', caa, gaa)/4
        #     + einsum('ijab,iajb', cbb, gbb)/4
        #     + einsum('ijab,iajb', cab, gab))
        if axis1 == 'fragment':
            assert len(c2) == 4
            caa, cab, cba, cbb = c2
            if c2ba_order == 'ab':
                cba = cba.transpose(1,0,3,2)
            e_doubles = (einsum('xi,xjab,iajb', pxa, caa, gaa)/4
                       - einsum('xi,xjab,ibja', pxa, caa, gaa)/4
                       + einsum('xi,xjab,iajb', pxb, cbb, gbb)/4
                       - einsum('xi,xjab,ibja', pxb, cbb, gbb)/4
                       + einsum('xi,xjab,iajb', pxa, cab, gab)/2
                       + einsum('xi,xjab,jbia', pxb, cba, gab)/2)
        else:
            assert len(c2) == 3
            caa, cab, cbb = c2
            e_doubles = (einsum('ijab,iajb', caa, gaa)/4
                       - einsum('ijab,ibja', caa, gaa)/4
                       + einsum('ijab,iajb', cbb, gbb)/4
                       - einsum('ijab,ibja', cbb, gbb)/4
                       + einsum('ijab,iajb', cab, gab))

        e_singles = (self.sym_factor * e_singles)
        e_doubles = (self.sym_factor * e_doubles)
        e_corr = (e_singles + e_doubles)
        return e_singles, e_doubles, e_corr

    def _get_projected_gamma1_intermediates(self, t_as_lambda=None, sym_t2=True):
        raise NotImplementedError
        t1, t2, l1, l2, t1x, t2x, l1x, l2x = self._ccsd_amplitudes_for_dm(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
        doo, dov, dvo, dvv = pyscf.cc.uccsd_rdm._gamma1_intermediates(None, t1, t2, l1x, l2x)
        # Correction for term without Lambda amplitude:
        #dvo += (t1x - t1).T
        #d1 = (doo, dov, dvo, dvv)
        #return d1

    def _get_projected_gamma2_intermediates(self, t_as_lambda=None, sym_t2=True):
        t1, t2, l1, l2, t1x, t2x, l1x, l2x = self._ccsd_amplitudes_for_dm(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
        # Only incore for UCCSD:
        #d2 = pyscf.cc.uccsd_rdm._gamma2_intermediates(None, t1, t2, l1x, l2x)
        d2ovov, *d2 = pyscf.cc.uccsd_rdm._gamma2_intermediates(None, t1, t2, l1x, l2x)
        # Correction of unprojected terms (which do not involve L1/L2):
        # dovov:
        dtau = (t2x[0]-t2[0] + einsum('ia,jb->ijab', t1x[0]-t1[0], 2*t1[0]))/4
        d2ovov[0][:] += dtau.transpose(0,2,1,3)
        d2ovov[0][:] -= dtau.transpose(0,3,1,2)
        # dovOV (symmetrize between t1x[0] and t1x[1]; t2x[1] should already be symmetrized):
        dtau = ((t2x[1]-t2[1]) + einsum('ia,jb->ijab', t1x[0]-t1[0], t1[1]/2)
                               + einsum('ia,jb->ijab', t1[0]/2, t1x[1]-t1[1]))/2
        d2ovov[1][:] += dtau.transpose(0,2,1,3)
        # dOVOV:
        dtau = (t2x[2]-t2[2] + einsum('ia,jb->ijab', t1x[1]-t1[1], 2*t1[1]))/4
        d2ovov[3][:] += dtau.transpose(0,2,1,3)
        d2ovov[3][:] -= dtau.transpose(0,3,1,2)
        d2 = (d2ovov, *d2)
        return d2

    def make_fragment_dm2cumulant(self, t_as_lambda=None, sym_t2=True, approx_cumulant=True, full_shape=True):
        """TODO: MP2"""
        if self.solver == 'MP2':
            raise NotImplementedError
        if (approx_cumulant not in (1, True)):
            raise NotImplementedError
        cc = d1 = None
        d2 = self._get_projected_gamma2_intermediates(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
        dm2 = pyscf.cc.uccsd_rdm._make_rdm2(cc, d1, d2, with_dm1=False, with_frozen=False)
        return dm2

    @log_method()
    def make_fragment_dm2cumulant_energy(self, t_as_lambda=False, sym_t2=True, approx_cumulant=True):
        dm2 = self.make_fragment_dm2cumulant(t_as_lambda=t_as_lambda, sym_t2=sym_t2, approx_cumulant=approx_cumulant,
                full_shape=False)
        #fac = (2 if self.solver == 'MP2' else 1)
        if self._eris is None:
            eris = self.base.get_eris_array(self.cluster.c_active)
        # CCSD
        elif hasattr(self._eris, 'ovoo'):
            #eris = vayesta.core.ao2mo.helper.get_full_array(self._eris)
            return vayesta.core.ao2mo.helper.contract_dm2_eris_uhf(dm2, self._eris)/2
        # MP2
        else:
            eris = self._eris
        dm2aa, dm2ab, dm2bb = dm2
        gaa, gab, gbb = eris
        e_dm2 = (einsum('ijkl,ijkl->', gaa, dm2aa)
               + einsum('ijkl,ijkl->', gab, dm2ab)*2
               + einsum('ijkl,ijkl->', gbb, dm2bb))/2
        return e_dm2
