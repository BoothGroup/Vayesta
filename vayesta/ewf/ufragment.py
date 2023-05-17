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

    def get_fragment_energy(self, c1, c2, hamil=None, fock=None, axis1='fragment', c2ba_order='ba'):
        """Calculate fragment correlation energy contribution from projected C1, C2.

        Parameters
        ----------
        c1: (n(occ-CO), n(vir-CO)) array
            Fragment projected C1-amplitudes.
        c2: (n(occ-CO), n(occ-CO), n(vir-CO), n(vir-CO)) array
            Fragment projected C2-amplitudes.
        hamil : ClusterHamiltonian object.
            Object representing cluster hamiltonian, possibly including cached ERIs.
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
            pxa, pxb = self.get_overlap('proj|cluster-occ')

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
        if hamil is None:
            hamil = self.hamil
        gaa = hamil.get_eris_bare(block="ovov")
        gab = hamil.get_eris_bare(block="ovOV")
        gbb = hamil.get_eris_bare(block="OVOV")

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

    @with_doc(RFragment._get_projected_gamma1_intermediates)
    def _get_projected_gamma1_intermediates(self, t_as_lambda=None, sym_t2=True):
        raise NotImplementedError

    @with_doc(RFragment._get_projected_gamma2_intermediates)
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
        if int(approx_cumulant) != 1:
            raise NotImplementedError

        if self.solver == 'MP2':
            t2xaa, t2xab, t2xbb = self.results.pwf.restore(sym=sym_t2).as_ccsd().t2
            dovov = t2xaa.transpose(0,2,1,3)
            dovOV = t2xab.transpose(0,2,1,3)
            dOVOV = t2xbb.transpose(0,2,1,3)
            if not full_shape:
                return (dovov, dovOV, dOVOV)
            nocca, nvira, noccb, nvirb = dovOV.shape
            norba = nocca + nvira
            norbb = noccb + nvirb
            oa, va = np.s_[:nocca], np.s_[nocca:]
            ob, vb = np.s_[:noccb], np.s_[noccb:]
            dm2aa = np.zeros(4*[norba])
            dm2ab = np.zeros(2*[norba] + 2*[norbb])
            dm2bb = np.zeros(4*[norbb])
            dm2aa[oa,va,oa,va] = dovov
            dm2aa[va,oa,va,oa] = dovov.transpose(1,0,3,2)
            dm2ab[oa,va,ob,vb] = dovOV
            dm2ab[va,oa,vb,ob] = dovOV.transpose(1,0,3,2)
            dm2bb[ob,vb,ob,vb] = dOVOV
            dm2bb[vb,ob,vb,ob] = dOVOV.transpose(1,0,3,2)
            return (dm2aa, dm2ab, dm2bb)

        cc = d1 = None
        d2 = self._get_projected_gamma2_intermediates(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
        dm2 = pyscf.cc.uccsd_rdm._make_rdm2(cc, d1, d2, with_dm1=False, with_frozen=False)
        return dm2

    @log_method()
    def make_fragment_dm2cumulant_energy(self, hamil=None, t_as_lambda=None, sym_t2=True, approx_cumulant=True):
        if hamil is None:
            hamil = self.hamil
        if self.solver == "MP2":
            dm2 = self.make_fragment_dm2cumulant(t_as_lambda=t_as_lambda, sym_t2=sym_t2,
                                                 approx_cumulant=approx_cumulant)
            dm2aa, dm2ab, dm2bb = dm2
            gaa = hamil.get_eris_bare(block="ovov")
            gab = hamil.get_eris_bare(block="ovOV")
            gbb = hamil.get_eris_bare(block="OVOV")
            return 2.0 * (einsum('ijkl,ijkl->', gaa, dm2aa)
                          + einsum('ijkl,ijkl->', gab, dm2ab) * 2
                          + einsum('ijkl,ijkl->', gbb, dm2bb)) / 2
        elif approx_cumulant:
            # Working hypothesis: this branch will effectively always uses `approx_cumulant=True`.
            eris = hamil.get_dummy_eri_object(force_bare=True)
            d2 = self._get_projected_gamma2_intermediates(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
            return vayesta.core.ao2mo.helper.contract_dm2intermeds_eris_uhf(d2, eris)/2
        else:
            dm2 = self.make_fragment_dm2cumulant(t_as_lambda=t_as_lambda, sym_t2=sym_t2,
                                                 approx_cumulant=approx_cumulant, full_shape=True)
            dm2aa, dm2ab, dm2bb = dm2
            gaa, gab, gbb = hamil.get_eris_bare()
            return (einsum('ijkl,ijkl->', gaa, dm2aa)
                    + einsum('ijkl,ijkl->', gab, dm2ab)*2
                    + einsum('ijkl,ijkl->', gbb, dm2bb))/2
