import numpy as np

from .qemb import QEmbedding
from .ufragment import UQEmbeddingFragment

class UQEmbedding(QEmbedding):
    """Spin unrestricted quantum embedding."""

    # Shadow this in inherited methods:
    Fragment = UQEmbeddingFragment

    def init_vhf(self):
        if self.opts.recalc_vhf:
            self.log.debug("Recalculating HF potential from MF object.")
            return None
        self.log.debug("Determining HF potential from MO energies and coefficients.")
        cs = einsum('...ai,ab->...ib', self.mo_coeff, self.get_ovlp())
        fock = einsum('...ia,...i,...ib->ab', cs, self.mo_energy, cs)
        return (fock - self.get_hcore())

    @property
    def nmo(self):
        """Total number of molecular orbitals (MOs)."""
        return (self.mo_coeff[0].shape[-1],
                self.mo_coeff[1].shape[-1])

    @property
    def nocc(self):
        """Number of occupied MOs."""
        return (np.count_nonzero(self.mo_occ[0] > 0),
                np.count_nonzero(self.mo_occ[1] > 0))

    @property
    def nvir(self):
        """Number of virtual MOs."""
        return (np.count_nonzero(self.mo_occ[0] == 0),
                np.count_nonzero(self.mo_occ[1] == 0))

    @property
    def mo_coeff_occ(self):
        """Occupied MO coefficients."""
        return (self.mo_coeff[0][:,:self.nocc[0]],
                self.mo_coeff[1][:,:self.nocc[1]])

    @property
    def mo_coeff_vir(self):
        """Virtual MO coefficients."""
        return (self.mo_coeff[0][:,self.nocc[0]:],
                self.mo_coeff[1][:,self.nocc[1]:])

    # TODO:

    def get_t1(self, *args, **kwargs):
        raise NotImplementedError()

    def get_t12(self, *args, **kwargs):
        raise NotImplementedError()

    def get_rdm1_demo(self, *args, **kwargs):
        raise NotImplementedError()

    def get_rdm1_ccsd(self, *args, **kwargs):
        raise NotImplementedError()

    def get_rdm2_demo(self, *args, **kwargs):
        raise NotImplementedError()

    def get_rdm2_ccsd(self, *args, **kwargs):
        raise NotImplementedError()

    def pop_analysis(self, *args, **kwargs):
        raise NotImplementedError()

UQEmbeddingMethod = UQEmbedding
