import numpy as np

from vayesta.core.util import *
from vayesta.core.qemb import UFragment
from vayesta.core.bath import UDMET_Bath, UCompleteBath
from .fragment import EWFFragment


class UEWFFragment(UFragment, EWFFragment):

    def set_cas(self, *args, **kwargs):
        raise NotImplementedError()

    def truncate_bno(self, c_no, n_no, *args, **kwargs):
        results = []
        for s, spin in enumerate(('alpha', 'beta')):
            self.log.info("%s:", spin.capitalize())
            results.append(super().truncate_bno(c_no[s], n_no[s], *args, **kwargs))
        return tuple(zip(*results))

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
            raise NotImplementedError()
            #bath = BNO_Bath(self, dmet_threshold=self.opts.dmet_threshold)
        else:
            raise ValueError("Unknown bath_type: %r" % bath_type)
        bath.kernel()
        self.bath = bath
        return bath

    def get_fragment_energy(self, c1, c2, eris, solver):
        if (self.opts.energy_factor*self.sym_factor) == 0: return 0

        nocc = (c1[0].shape[0], c1[1].shape[0])
        nvir = (c1[0].shape[1], c1[1].shape[1])
        oa, ob = np.s_[:nocc[0]], np.s_[:nocc[1]]
        va, vb = np.s_[nocc[0]:], np.s_[nocc[1]:]

        # Singles energy (for non-HF reference)
        ca, cb = c1
        if hasattr(eris, 'fock'):
            fa = eris.fock[0][oa,va]
            fb = eris.fock[1][ob,vb]
        else:
            fock = self.base.get_fock()
            fa = dot(self.c_active_occ[0].T, fock[0], self.c_active_vir[0])
            fb = dot(self.c_active_occ[1].T, fock[1], self.c_active_vir[1])
        e_s = np.sum(fa*ca) + np.sum(fb*cb)
        # Doubles energy
        # TODO: loop to reduce memory footprint
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

        caa, cab, cbb = c2
        #caa = caa + einsum('ia,jb->ijab', ca, ca) - einsum('ib,ja->ijab', ca, ca)
        #cbb = cbb + einsum('ia,jb->ijab', cb, cb) - einsum('ib,ja->ijab', cb, cb)
        #e_d = (einsum('ijab,iajb', caa, gaa)/4
        #     + einsum('ijab,iajb', cbb, gbb)/4
        #     + einsum('ijab,iajb', cab, gab))
        e_d = (einsum('ijab,iajb', caa, gaa)/4 - einsum('ijab,ibja', caa, gaa)/4
             + einsum('ijab,iajb', cbb, gbb)/4 - einsum('ijab,ibja', cbb, gbb)/4
             + einsum('ijab,iajb', cab, gab))

        self.log.debug("Energy components: E(singles)= %s, E(doubles)= %s",
                energy_string(e_s), energy_string(e_d))
        if solver != 'FCI' and (e_s > 0.1*e_d and e_s > 1e-4):
            self.log.warning("Large E(singles) component!")
        e_frag = self.opts.energy_factor * self.sym_factor * (e_s + e_d)
        return e_frag
