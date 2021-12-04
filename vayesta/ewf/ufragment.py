import numpy as np

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
            bath = UMP2_BNO_Bath(self, dmet_threshold=self.opts.dmet_threshold)
        else:
            raise ValueError("Unknown bath_type: %r" % bath_type)
        bath.kernel()
        self.bath = bath
        return bath

    def get_fragment_energy(self, c1, c2, eris, fock=None, axis1='cluster'):
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

        nocc = (c1[0].shape[0], c1[1].shape[0])
        nvir = (c1[0].shape[1], c1[1].shape[1])
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
        else:
            gaa = eris[0][oa,va,oa,va]
            gab = eris[1][oa,va,ob,vb]
            gbb = eris[2][ob,vb,ob,vb]

        #caa = caa + einsum('ia,jb->ijab', ca, ca) - einsum('ib,ja->ijab', ca, ca)
        #cbb = cbb + einsum('ia,jb->ijab', cb, cb) - einsum('ib,ja->ijab', cb, cb)
        #e_d = (einsum('ijab,iajb', caa, gaa)/4
        #     + einsum('ijab,iajb', cbb, gbb)/4
        #     + einsum('ijab,iajb', cab, gab))
        if axis1 == 'fragment':
            caa, cab, cba, cbb = c2
            e_doubles = (einsum('xi,xjab,iajb', pxa, caa, gaa)/4 - einsum('xi,xjab,ibja', pxa, caa, gaa)/4
                       + einsum('xi,xjab,iajb', pxb, cbb, gbb)/4 - einsum('xi,xjab,ibja', pxb, cbb, gbb)/4
                       + einsum('xi,xjab,iajb', pxa, cab, gab)/2
                       + einsum('xj,ixab,iajb', pxb, cba, gab)/2)
        else:
            caa, cab, cbb = c2
            e_doubles = (einsum('ijab,iajb', caa, gaa)/4 - einsum('ijab,ibja', caa, gaa)/4
                       + einsum('ijab,iajb', cbb, gbb)/4 - einsum('ijab,ibja', cbb, gbb)/4
                       + einsum('ijab,iajb', cab, gab))

        e_singles = (self.opts.energy_factor*self.sym_factor * e_singles)
        e_doubles = (self.opts.energy_factor*self.sym_factor * e_doubles)
        e_corr = (e_singles + e_doubles)
        return e_singles, e_doubles, e_corr
