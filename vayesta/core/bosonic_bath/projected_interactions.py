import numpy as np
from vayesta.core.util import einsum
import pyscf.lib

class BosonicHamiltonianProjector:
    def __init__(self, cluster, mo_cderi_getter, mf):
        self.cluster = cluster
        if cluster.bosons is None:
            raise ValueError("Cluster has no defined bosons to generate interactions for!")
        # Functions to get
        self.mo_cderi_getter = mo_cderi_getter
        self.mf = mf
        assert(self.cluster.inc_bosons)
        self._cderi_clus = None
        self._cderi_bos = None

    @property
    def bcluster(self):
        return self.cluster.bosons

    @property
    def qba_basis(self):
        return self.bcluster.forbitals

    @property
    def cderi_clus(self):
        if self._cderi_clus is None:
            c_active = self.cluster.c_active
            if c_active[0].ndim == 1:
                cderi = self.mo_cderi_getter(c_active)
                self._cderi_clus = (cderi, cderi)
            else:
                self._cderi_clus = (self.mo_cderi_getter(c_active[0]),
                                    self.mo_cderi_getter(c_active[1]))
        return self._cderi_clus

    @property
    def cderi_bos(self):
        if self._cderi_bos is None:
            rbos = sum(self.bcluster.coeff_3d_ao)
            cderi = np.zeros((self.naux, self.bcluster.nbos))
            cderi_neg = None
            for blk, lab in self._loop_df():
                if blk is not None:
                    cderi[blk] = einsum('Lab,nab->Ln', lab, rbos)
                else:
                    cderi_neg = einsum('Lab,nab->Ln', lab, rbos)
            self._cderi_bos = (cderi, cderi_neg)
        return self._cderi_bos

    @property
    def naux(self):
        df = self.mf.with_df
        try:
            return (df.auxcell.nao if hasattr(df, 'auxcell') else df.auxmol.nao)
        except AttributeError:
            return df.get_naoaux()

    def _loop_df(self, blksize=None):
        nao = self.mf.mol.nao
        df = self.mf.with_df
        naux = self.naux
        if blksize is None:
            blksize = int(1e9 / naux * nao * nao * 8)
        # PBC:
        if hasattr(df, 'sr_loop'):
            blk0 = 0
            for labr, labi, sign in df.sr_loop(compact=False, blksize=blksize):
                assert np.allclose(labi, 0)
                labr = labr.reshape(-1, nao, nao)
                if (sign == 1):
                    blk1 = (blk0 + labr.shape[0])
                    blk = np.s_[blk0:blk1]
                    blk0 = blk1
                    yield blk, labr
                elif (sign == -1):
                    yield None, labr
        # No PBC:
        blk0 = 0
        for lab in df.loop(blksize=blksize):
            blk1 = (blk0 + lab.shape[0])
            blk = np.s_[blk0:blk1]
            blk0 = blk1
            lab = pyscf.lib.unpack_tril(lab)
            yield blk, lab

    def project_hamiltonian(self, coupling_exchange=False, freq_exchange=False):
        freqs, c = self.project_freqs(exchange=freq_exchange)
        couplings = self.project_couplings(exchange=coupling_exchange)
        return freqs, einsum("nm,npq->mpq", c, couplings), c

    def project_couplings(self, exchange=False):
        if exchange:
            raise NotImplementedError
        # For coulombic contributions we just need these cderis.
        cderi_clus, cderi_clus_neg = self.cderi_clus
        cderi_bos, cderi_bos_neg = self.cderi_bos
        couplings = einsum("Ln,Lpq->npq", cderi_bos, cderi_clus)
        if cderi_clus_neg is not None:
            if cderi_bos_neg is None:
                raise ValueError("Only have negative cderi contribution via one channel; something's gone wrong.")
            couplings -= einsum("Ln,Lpq->npq", cderi_bos_neg, cderi_clus_neg)
        return couplings

    def project_freqs(self, exchange=False):
        if exchange:
            raise NotImplementedError
        cderi_bos, cderi_bos_neg = self.cderi_bos
        hbb = einsum("Ln,Lm->nm", cderi_bos, cderi_bos)
        # Want to take eigenvectors of this coupling matrix as our bosonic auxiliaries.
        freqs, c = np.linalg.eigh(hbb)
        return freqs, c




