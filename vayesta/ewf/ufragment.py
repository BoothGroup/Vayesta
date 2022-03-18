import numpy as np

import pyscf
import pyscf.cc

from vayesta.core.util import *
from vayesta.core.qemb import UFragment
from vayesta.core.bath import UDMET_Bath
from vayesta.core.bath import UCompleteBath
from vayesta.core.bath import UMP2_BNO_Bath
from .fragment import EWFFragment


class UEWFFragment(UFragment, EWFFragment):

    def set_cas(self, *args, **kwargs):
        raise NotImplementedError()

    def make_bath(self, bath_type=NotSet):
        if bath_type is NotSet:
            bath_type = self.opts.bath_type
        # DMET bath only
        if bath_type is None or bath_type.lower() == 'dmet':
            bath = UDMET_Bath(self, dmet_threshold=self.opts.dmet_threshold)
        # All environment orbitals as bath
        elif bath_type.lower() in ('all', 'full'):
            #raise NotImplementedError()
            bath = UCompleteBath(self, dmet_threshold=self.opts.dmet_threshold)
        # MP2 bath natural orbitals
        elif bath_type.lower() == 'mp2-bno':
            dmet_bath = UDMET_Bath(self, dmet_threshold=self.opts.dmet_threshold)
            dmet_bath.kernel()
            bath = UMP2_BNO_Bath(self, dmet_bath)
        else:
            raise ValueError("Unknown bath_type: %r" % bath_type)
        bath.kernel()
        self.bath = bath
        return bath

    def get_fragment_energy(self, c1, c2, eris, fock=None, axis1='fragment', c2ba_order='ba'):
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
        if (self.opts.energy_factor*self.sym_factor) == 0: return 0

        nocc = (c2[0].shape[1], c2[-1].shape[1])
        nvir = (c2[0].shape[2], c2[-1].shape[2])
        self.log.debugv("nocc= %d, %d nvir= %d, %d", *nocc, *nvir)
        oa, ob = np.s_[:nocc[0]], np.s_[:nocc[1]]
        va, vb = np.s_[nocc[0]:], np.s_[nocc[1]:]
        if axis1 == 'fragment':
            pxa, pxb = self.get_occ2frag_projector()

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

        e_singles = (self.opts.energy_factor*self.sym_factor * e_singles)
        e_doubles = (self.opts.energy_factor*self.sym_factor * e_doubles)
        e_corr = (e_singles + e_doubles)
        return e_singles, e_doubles, e_corr

    def make_partial_dm2(self, t_as_lambda=False, sym_t2=True):
        t1 = self.results.wf.t1
        t2 = self.results.wf.t2
        pwf = self.results.pwf.restore(sym=sym_t2)
        t1x, t2x = pwf.t1, pwf.t2
        if t_as_lambda:
            l1x, l2x = t1x, t2x
        else:
            l1x, l2x = pwf.l1, pwf.l2

        # Only incore for UCCSD:
        d2 = pyscf.cc.uccsd_rdm._gamma2_intermediates(None, t1, t2, l1x, l2x)

        # Correction of unprojected terms (which do not involve L1/L2):
        # dovov:
        dtau = (t2x[0]-t2[0] + einsum('ia,jb->ijab', t1x[0]-t1[0], 2*t1[0]))/4
        d2[0][0][:] += dtau.transpose(0,2,1,3)
        d2[0][0][:] -= dtau.transpose(0,3,1,2)
        # dovOV (symmetrize between t1x[0] and t1x[1]; t2x[1] should already be symmetrized):
        dtau = ((t2x[1]-t2[1]) + einsum('ia,jb->ijab', t1x[0]-t1[0], t1[1]/2)
                               + einsum('ia,jb->ijab', t1[0]/2, t1x[1]-t1[1]))/2
        d2[0][1][:] += dtau.transpose(0,2,1,3)
        # dOVOV:
        dtau = (t2x[2]-t2[2] + einsum('ia,jb->ijab', t1x[1]-t1[1], 2*t1[1]))/4
        d2[0][3][:] += dtau.transpose(0,2,1,3)
        d2[0][3][:] -= dtau.transpose(0,3,1,2)
        dm2 = pyscf.cc.uccsd_rdm._make_rdm2(None, None, d2, with_dm1=False, with_frozen=False)
        return dm2

    def get_cluster_sz(self, proj=None):
        """<P S_z>"""
        dm1 = self.results.dm1
        if dm1 is None:
            raise ValueError()
        dm1a, dm1b = dm1

        if proj is None:
            sz = (einsum('ii->', dm1a) - einsum('ii->', dm1b))/2
            return sz

        def get_proj_per_spin(p):
            if np.ndim(p[0]) == 2:
                return p
            if np.ndim(p[0]) == 1:
                return p, p
            raise ValueError()

        proja, projb = get_proj_per_spin(proj)
        sz = (einsum('ij,ij->', dm1a, proja)
            - einsum('ij,ij->', dm1b, projb))/2
        return sz

    def get_cluster_ssz(self, proj1=None, proj2=None):
        """<P1 S_z P2 S_z>"""
        dm1 = self.results.dm1
        dm2 = self.results.dm2
        if dm1 is None or dm2 is None:
            raise ValueError()
        dm1a, dm1b = dm1
        dm2aa, dm2ab, dm2bb = dm2

        if proj1 is None:
            ssz = (einsum('iijj->', dm2aa)/4 + einsum('iijj->', dm2bb)/4
                 - einsum('iijj->', dm2ab)/2)
            ssz += (einsum('ii->', dm1a) + einsum('ii->', dm1b))/4
            return ssz

        def get_proj_per_spin(p):
            if np.ndim(p[0]) == 2:
                return p
            if np.ndim(p[0]) == 1:
                return p, p
            raise ValueError()

        if proj2 is None:
            proj2 = proj1
        proj1a, proj1b = get_proj_per_spin(proj1)
        proj2a, proj2b = get_proj_per_spin(proj2)
        ssz = (einsum('ijkl,ij,kl->', dm2aa, proj1a, proj2a)/4
             + einsum('ijkl,ij,kl->', dm2bb, proj1b, proj2b)/4
             - einsum('ijkl,ij,kl->', dm2ab, proj1a, proj2b)/4
             - einsum('ijkl,ij,kl->', dm2ab, proj2a, proj1b)/4)
        ssz += (einsum('ij,ik,jk->', dm1a, proj1a, proj2a)
              + einsum('ij,ik,jk->', dm1b, proj1b, proj2b))/4
        return ssz
