import numpy as np

from vayesta.core.types import MP2_WaveFunction
from vayesta.core.util import *
from .solver import ClusterSolver, UClusterSolver

import dataclasses


class RMP2_Solver(ClusterSolver):
    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        compress_cderi: bool = False

    def kernel(self, *args, **kwargs):

        nao, mo_coeff, mo_energy, mo_occ, ovlp = self.hamil.get_clus_mf_info(ao_basis=False)

        eris = cderi = cderi_neg = None
        if not self.hamil.has_screening:
            try:
                cderi, cderi_neg = self.hamil.get_cderi_bare(only_ov=True, compress=self.opts.compress_cderi)
            except AttributeError:
                cderi = cderi_neg = None

        if cderi is None:
            cderi = cderi_neg = None
            nocc = sum(mo_occ.T > 0)
            eris = self.hamil.get_eris()
            eris = self.get_ov_eris(eris, nocc)
        with log_time(self.log.timing, "Time for MP2 T-amplitudes: %s"):
            t2 = self.make_t2(mo_energy, eris=eris, cderi=cderi, cderi_neg=cderi_neg)
        self.wf = MP2_WaveFunction(self.hamil.mo, t2)
        self.converged = True

    def get_ov_eris(self, eris, nocc):
        return eris[:nocc, nocc:, :nocc, nocc:]

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
        eia = (mo_energy[:nocc, None] - mo_energy[None, nocc:])
        if blksize is None:
            blksize = int(1e9 / max(nocc * nvir * nvir * 8, 1))
        for blk in brange(0, nocc, blksize):
            if eris is not None:
                gijab = eris[blk].transpose(0, 2, 1, 3)
            else:
                gijab = einsum('Lia,Ljb->ijab', cderi[:, blk], cderi)
                if cderi_neg is not None:
                    gijab -= einsum('Lia,Ljb->ijab', cderi_neg[:, blk], cderi_neg)
            eijab = (eia[blk][:, None, :, None] + eia[None, :, None, :])
            t2[blk] = (gijab / eijab)
        return t2

    def _debug_exact_wf(self, wf):
        raise NotImplementedError

    def _debug_random_wf(self):
        mo = self.hamil.mo
        t2 = np.random.rand(mo.nocc, mo.nocc, mo.nvir, mo.nvir)
        self.wf = MP2_WaveFunction(mo, t2)
        self.converged = True


class UMP2_Solver(UClusterSolver, RMP2_Solver):

    def get_ov_eris(self, eris, nocc):
        na, nb = nocc
        gaa, gab, gbb = eris
        return gaa[:na, na:, :na, na:], gab[:na, na:, :nb, nb:], gbb[:nb, nb:, :nb, nb:]

    def make_t2(self, mo_energy, eris=None, cderi=None, cderi_neg=None, blksize=None, workmem=int(1e9)):
        """Make T2 amplitudes"""
        # (ov|ov)
        if eris is not None:
            self.log.debugv("Making T2 amplitudes from ERIs")
            assert len(eris) == 3
            assert (eris[0].ndim == 4)
            assert (eris[1].ndim == 4)
            assert (eris[2].ndim == 4)
            nocca, nvira = eris[0].shape[:2]
            noccb, nvirb = eris[2].shape[:2]
        # (L|ov)
        elif cderi is not None:
            self.log.debugv("Making T2 amplitudes from CD-ERIs")
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
        eia_a = (mo_energy[0][:nocca, None] - mo_energy[0][None, nocca:])
        eia_b = (mo_energy[1][:noccb, None] - mo_energy[1][None, noccb:])

        # Alpha-alpha and Alpha-beta:
        if blksize is None:
            blksize_a = int(workmem / max(nocca * nvira * nvira * 8, 1))
        else:
            blksize_a = blksize
        for blk in brange(0, nocca, blksize_a):
            # Alpha-alpha
            if eris is not None:
                gijab = eris[0][blk].transpose(0, 2, 1, 3)
            else:
                gijab = einsum('Lia,Ljb->ijab', cderi[0][:, blk], cderi[0])
                if cderi_neg[0] is not None:
                    gijab -= einsum('Lia,Ljb->ijab', cderi_neg[0][:, blk], cderi_neg[0])
            eijab = (eia_a[blk][:, None, :, None] + eia_a[None, :, None, :])
            t2aa[blk] = (gijab / eijab)
            t2aa[blk] -= t2aa[blk].transpose(0, 1, 3, 2)
            # Alpha-beta
            if eris is not None:
                gijab = eris[1][blk].transpose(0, 2, 1, 3)
            else:
                gijab = einsum('Lia,Ljb->ijab', cderi[0][:, blk], cderi[1])
                if cderi_neg[0] is not None:
                    gijab -= einsum('Lia,Ljb->ijab', cderi_neg[0][:, blk], cderi_neg[1])
            eijab = (eia_a[blk][:, None, :, None] + eia_b[None, :, None, :])
            t2ab[blk] = (gijab / eijab)
        # Beta-beta:
        if blksize is None:
            blksize_b = int(workmem / max(noccb * nvirb * nvirb * 8, 1))
        else:
            blksize_b = blksize
        for blk in brange(0, noccb, blksize_b):
            if eris is not None:
                gijab = eris[2][blk].transpose(0, 2, 1, 3)
            else:
                gijab = einsum('Lia,Ljb->ijab', cderi[1][:, blk], cderi[1])
                if cderi_neg[0] is not None:
                    gijab -= einsum('Lia,Ljb->ijab', cderi_neg[1][:, blk], cderi_neg[1])
            eijab = (eia_b[blk][:, None, :, None] + eia_b[None, :, None, :])
            t2bb[blk] = (gijab / eijab)
            t2bb[blk] -= t2bb[blk].transpose(0, 1, 3, 2)

        return (t2aa, t2ab, t2bb)
