import numpy as np

from .solver import ClusterSolver

from vayesta.core.util import *
from vayesta.core.types import Orbitals
from vayesta.core.types import MP2_WaveFunction

class MP2_Solver(ClusterSolver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        frozen = self.cluster.get_frozen_indices()
        # --- Results
        self.t2 = None

    def reset(self):
        super().reset()
        self.t2 = None

    @deprecated()
    def get_t1(self):
        #return np.zeros((self.cluster.nocc_active, self.cluster.nvir_active))
        return None

    @deprecated()
    def get_c1(self, intermed_norm=True):
        #return self.get_t1()
        return None

    @deprecated()
    def get_t2(self):
        return self.t2

    @deprecated()
    def get_c2(self, intermed_norm=True):
        """C2 in intermediate normalization."""
        if not intermed_norm:
            raise ValueError()
        return self.t2

    def get_init_guess(self):
        return {}

    def get_eris(self):
        # We only need the (ov|ov) block for MP2:
        mo_coeff = 2*[self.cluster.c_active_occ, self.cluster.c_active_vir]
        with log_time(self.log.timing, "Time for 2e-integral transformation: %s"):
            eris = self.base.get_eris_array(mo_coeff)
        return eris

    def get_cderi(self):
        # We only need the (L|ov) block for MP2:
        mo_coeff = (self.cluster.c_active_occ, self.cluster.c_active_vir)
        with log_time(self.log.timing, "Time for 2e-integral transformation: %s"):
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

    #def _project_out_subspace(self, t2, c_occ, c_vir):
    #    """Alternative way to implement double counting correction."""
    #    ovlp = self.base.get_ovlp()
    #    r_occ = dot(self.cluster.c_active_occ.T, ovlp, c_occ)
    #    r_vir = dot(self.cluster.c_active_vir.T, ovlp, c_vir)
    #    p_occ = np.dot(r_occ, r_occ.T)
    #    p_vir = np.dot(r_vir, r_vir.T)
    #    t2_proj = einsum('IJAB,Ii,Jj,Aa,Bb->ijab', t2, p_occ, p_occ, p_vir, p_vir)
    #    return (t2 - t2_proj)

    def kernel(self, eris=None):

        if self.v_ext is not None:
            raise NotImplementedError

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
        # Alternative double counting correction for secondary fragments:
        #if self.fragment.flags.secondary_fragment:
        #    f = self.fragment.flags.bath_parent_fragment
        #    c_sub_occ = f.cluster.c_active_occ
        #    c_sub_vir = f.cluster.c_active_vir
        #    t2 = self._project_out_subspace(t2, c_sub_occ, c_sub_vir)

        self.t2 = t2
        mo = Orbitals(self.cluster.c_active, energy=mo_energy, occ=self.cluster.nocc_active)
        self.wf = MP2_WaveFunction(mo, t2)
        self.converged = True

    def _debug_exact_wf(self, wf):
        raise NotImplementedError

    def _debug_random_wf(self):
        mo = Orbitals(self.cluster.c_active, occ=self.cluster.nocc_active)
        t2 = np.random.rand(mo.nocc, mo.nocc, mo.nvir, mo.nvir)
        self.wf = MP2_WaveFunction(mo, t2)
        self.converged = True


class UMP2_Solver(MP2_Solver):

    def get_eris(self):
        # We only need the (ov|ov), (ov|OV) and (OV|OV) blocks for MP2:
        return self.base.get_eris_array_uhf(self.cluster.c_active_occ, mo_coeff2=self.cluster.c_active_vir)

    def get_cderi(self):
        # We only need the (L|ov) and (L|OV) blocks for MP2:
        c_aa = [self.cluster.c_active_occ[0], self.cluster.c_active_vir[0]]
        c_bb = [self.cluster.c_active_occ[1], self.cluster.c_active_vir[1]]
        cderi_a, cderi_neg_a = self.base.get_cderi(c_aa)
        cderi_b, cderi_neg_b = self.base.get_cderi(c_bb)
        cderi = (cderi_a, cderi_b)
        cderi_neg = (cderi_neg_a, cderi_neg_b)
        return cderi, cderi_neg

    def get_mo_energy(self, fock=None):
        if fock is None:
            fock = self.base.get_fock()
        c_act = self.cluster.c_active
        mo_energy_a = einsum('ai,ab,bi->i', c_act[0], fock[0], c_act[0])
        mo_energy_b = einsum('ai,ab,bi->i', c_act[1], fock[1], c_act[1])
        return (mo_energy_a, mo_energy_b)

    #def make_t2(self, mo_energy, eris=None, cderi=None, cderi_neg=None, blksize=None):
    #    """Make T2 amplitudes"""
    #    # (ov|ov)
    #    if eris is not None:
    #        self.log.debugv("Making T2 amplitudes from ERIs")
    #        assert (len(eris) == 3)
    #        assert (eris[0].ndim == 4)
    #        assert (eris[1].ndim == 4)
    #        assert (eris[2].ndim == 4)
    #        nocca, nvira, noccb, nvirb = eris[1].shape
    #    # (L|ov)
    #    elif cderi is not None:
    #        self.log.debugv("Making T2 amplitudes from CD-ERIs")
    #        assert (len(cderi) == 2)
    #        assert (cderi[0].ndim == 3)
    #        assert (cderi_neg[0] is None or cderi_neg[0].ndim == 3)
    #        nocca, nvira = cderi[0].shape[1:]
    #        noccb, nvirb = cderi[1].shape[1:]
    #    else:
    #        raise ValueError()

    #    t2aa = np.empty((nocca, nocca, nvira, nvira))
    #    t2ab = np.empty((nocca, noccb, nvira, nvirb))
    #    t2bb = np.empty((noccb, noccb, nvirb, nvirb))
    #    eia_a = (mo_energy[0][:nocca,None] - mo_energy[0][None,nocca:])
    #    eia_b = (mo_energy[0][:nocca,None] - mo_energy[0][None,nocca:])
    #    if blksize is None:
    #        blksize = int(1e9 / max(nocc*nvir*nvir * 8, 1))
    #    for blk in brange(0, nocc, blksize):
    #        if eris is not None:
    #            gijab = eris[blk].transpose(0,2,1,3)
    #        else:
    #            gijab = einsum('Lia,Ljb->ijab', cderi[:,blk], cderi)
    #            if cderi_neg is not None:
    #                gijab -= einsum('Lia,Ljb->ijab', cderi_neg[:,blk], cderi_neg)
    #        eijab = (eia[blk][:,None,:,None] + eia[None,:,None,:])
    #        t2[blk] = (gijab / eijab)
    #    return t2

    def make_t2(self, mo_energy, eris=None, cderi=None, cderi_neg=None, blksize=None, workmem=int(1e9)):
        """Make T2 amplitudes"""
        # (ov|ov)
        if eris is not None:
            assert len(eris) == 3
            assert (eris[0].ndim == 4)
            assert (eris[1].ndim == 4)
            assert (eris[2].ndim == 4)
            nocca, nvira = eris[0].shape[:2]
            noccb, nvirb = eris[2].shape[:2]
        # (L|ov)
        elif cderi is not None:
            assert len(cderi) == 2
            assert (cderi[0].ndim == 3)
            assert (cderi[1].ndim == 3)
            nocca, nvira = cderi[0].shape[1:]
            noccb, nvirb = cderi[1].shape[1:]
        else:
            raise ValueError()

        t2aa = np.empty((nocca, nocca, nvira, nvira))
        t2ab = np.empty((nocca, noccb, nvira, nvirb))
        t2bb = np.empty((noccb, noccb, nvirb, nvirb))
        eia_a = (mo_energy[0][:nocca,None] - mo_energy[0][None,nocca:])
        eia_b = (mo_energy[1][:noccb,None] - mo_energy[1][None,noccb:])

        # Alpha-alpha and Alpha-beta:
        if blksize is None:
            blksize_a = int(workmem / max(nocca*nvira*nvira * 8, 1))
        else:
            blksize_a = blksize
        for blk in brange(0, nocca, blksize_a):
            # Alpha-alpha
            if eris is not None:
                gijab = eris[0][blk].transpose(0,2,1,3)
            else:
                gijab = einsum('Lia,Ljb->ijab', cderi[0][:,blk], cderi[0])
                if cderi_neg[0] is not None:
                    gijab -= einsum('Lia,Ljb->ijab', cderi_neg[0][:,blk], cderi_neg[0])
            eijab = (eia_a[blk][:,None,:,None] + eia_a[None,:,None,:])
            t2aa[blk] = (gijab / eijab)
            t2aa[blk] -= t2aa[blk].transpose(0,1,3,2)
            # Alpha-beta
            if eris is not None:
                gijab = eris[1][blk].transpose(0,2,1,3)
            else:
                gijab = einsum('Lia,Ljb->ijab', cderi[0][:,blk], cderi[1])
                if cderi_neg[0] is not None:
                    gijab -= einsum('Lia,Ljb->ijab', cderi_neg[0][:,blk], cderi_neg[1])
            eijab = (eia_a[blk][:,None,:,None] + eia_b[None,:,None,:])
            t2ab[blk] = (gijab / eijab)
        # Beta-beta:
        if blksize is None:
            blksize_b = int(workmem / max(noccb*nvirb*nvirb * 8, 1))
        else:
            blksize_b = blksize
        for blk in brange(0, noccb, blksize_b):
            if eris is not None:
                gijab = eris[2][blk].transpose(0,2,1,3)
            else:
                gijab = einsum('Lia,Ljb->ijab', cderi[1][:,blk], cderi[1])
                if cderi_neg[0] is not None:
                    gijab -= einsum('Lia,Ljb->ijab', cderi_neg[1][:,blk], cderi_neg[1])
            eijab = (eia_b[blk][:,None,:,None] + eia_b[None,:,None,:])
            t2bb[blk] = (gijab / eijab)
            t2bb[blk] -= t2bb[blk].transpose(0,1,3,2)

        return (t2aa, t2ab, t2bb)
