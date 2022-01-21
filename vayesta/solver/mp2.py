import numpy as np

from .solver2 import ClusterSolver

from vayesta.core.util import *

class MP2_Solver(ClusterSolver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        frozen = self.cluster.get_frozen_indices()
        # --- Results
        self.t2 = None

    def reset(self):
        super().reset()
        self.t2 = None

    def get_t1(self):
        #return np.zeros((self.cluster.nocc_active, self.cluster.nvir_active))
        return None

    def get_c1(self, intermed_norm=True):
        #return self.get_t1()
        return None

    def get_t2(self):
        return self.t2

    def get_c2(self, intermed_norm=True):
        """C2 in intermediate normalization."""
        if not intermed_norm:
            raise ValueError()
        return self.t2

    def get_eris(self):
        # We only need the (ov|ov) block for MP2:
        mo_coeff = 2*[self.cluster.c_active_occ, self.cluster.c_active_vir]
        eris = self.base.get_eris_array(mo_coeff)
        return eris

    def get_cderi(self):
        # We only need the (L|ov) block for MP2:
        mo_coeff = (self.cluster.c_active_occ, self.cluster.c_active_vir)
        cderi, cderi_neg = self.base.get_cderi(mo_coeff)
        return cderi, cderi_neg

    def get_mo_energy(self, fock=None):
        if fock is None:
            fock = self.base.get_fock()
        c_act = self.cluster.c_active
        mo_energy = einsum('ai,ab,bi->i', c_act, fock, c_act)
        return mo_energy

    def make_t2(self, mo_energy, eris=None, cderi=None, cderi_neg=None, blksize=None):
        """Make T2 amplitudes"""
        # (ov|ov)
        if eris is not None:
            self.log.debugv("Making T2 amplitudes from ERIs")
            assert (eris.ndim == 4)
            nocc, nvir = eris.shape[:2]
        # (L|ov)
        elif cderi is not None:
            self.log.debugv("Making T2 amplitudes from CD-ERIs")
            assert (cderi.ndim == 3)
            assert (cderi_neg is None or cderi_neg.ndim == 3)
            nocc, nvir = cderi.shape[1:]
        else:
            raise ValueError()

        t2 = np.empty((nocc, nocc, nvir, nvir))
        eia = (mo_energy[:nocc,None] - mo_energy[None,nocc:])
        if blksize is None:
            blksize = int(1e9 / max(nocc*nvir*nvir * 8, 1))
        for blk in brange(0, nocc, blksize):
            if eris is not None:
                gijab = eris[blk].transpose(0,2,1,3)
            else:
                gijab = einsum('Lia,Ljb->ijab', cderi[:,blk], cderi)
                if cderi_neg is not None:
                    gijab -= einsum('Lia,Ljb->ijab', cderi_neg[:,blk], cderi_neg)
            eijab = (eia[blk][:,None,:,None] + eia[None,:,None,:])
            t2[blk] = (gijab / eijab)
        return t2

    def kernel(self, eris=None):
        if eris is None:
            eris = cderi = cderi_neg = None
            with log_time(self.log.timing, "Time for AO->MO transformation: %s"):
                if self.base.has_df:
                    cderi, cderi_neg = self.get_cderi()
                else:
                    eris = self.get_eris()
        else:
            cderi = cderi_neg = None

        mo_energy = self.get_mo_energy()
        with log_time(self.log.timing, "Time for MP2 T-amplitudes: %s"):
            t2 = self.make_t2(mo_energy, eris=eris, cderi=cderi, cderi_neg=cderi_neg)
        self.t2 = t2
