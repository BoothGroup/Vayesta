import dataclasses

import numpy as np

from vayesta.core.util import OptionsBase
from . import ebfci_slow, uebfci_slow


class REBFCI:
    """Performs FCI on coupled electron-boson systems.
    Input:
        -system specification, via hamiltonian object along with bosonic parameters.
    Output:
        -FCI RDMs.
        -FCI response specification.
    """
    solver = ebfci_slow

    @dataclasses.dataclass
    class Options(OptionsBase):
        max_boson_occ: int = 2
        conv_tol: float = 1e-12
        max_cycle: int = 100

    def __init__(self, hamil, freqs, couplings, **kwargs):
        self.hamil = hamil
        self.freqs = freqs
        self.couplings = couplings
        self.opts = self.Options().replace(**kwargs)

    @property
    def norb(self):
        return self.hamil.ncas[0]

    @property
    def nelec(self):
        return self.hamil.nelec

    @property
    def nbos(self):
        return len(self.hamil.bos_freqs)

    def kernel(self, eris=None):
        # Get MO eris.
        h1e, eris = self.get_hamil(eris)
        couplings = self.hamil.couplings
        bos_freqs = self.hamil.bos_freqs
        self.e_fci, self.civec = self.solver.kernel(h1e, eris, couplings, np.diag(bos_freqs), self.norb, self.nelec,
                                                    self.nbos, max_occ=self.opts.max_boson_occ, tol=self.opts.conv_tol,
                                                    max_cycle=self.opts.max_cycle)
        return self.e_fci, self.civec

    def get_hamil(self, eris=None):
        h1e = self.hamil.get_heff(eris)
        eris = self.hamil.get_eris_screened()
        return h1e, eris

    def make_rdm1(self):
        return self.solver.make_rdm1(self.civec, self.norb, self.nelec)

    def make_rdm2(self):
        dm1, dm2 = self.make_rdm12()
        return dm2

    def make_rdm12(self):
        return self.solver.make_rdm12(self.civec, self.norb, self.nelec)

    def make_rdm_eb(self):
        # Note this is always spin-resolved, since bosonic couplings can have spin-dependence.
        return self.solver.make_eb_rdm(self.civec, self.norb, self.nelec, self.nbos, self.opts.max_boson_occ)

    def make_dd_moms(self, max_mom, dm1=None, coeffs=None, civec=None, eris=None):
        if civec is None:
            civec = self.civec
        h1e, eris = self.get_hamil(eris)

        if dm1 is None:
            dm1 = self.make_rdm1()

        self.dd_moms = self.solver.calc_dd_resp_mom(
            civec, self.e_fci, max_mom, self.norb, self.nelec, self.nbos, h1e, eris,
            np.diag(self.freqs), self.couplings, self.opts.max_boson_occ, dm1,
            coeffs=coeffs)
        return self.dd_moms


class UEBFCI(REBFCI):
    solver = uebfci_slow

    def make_rdm12(self):
        return self.solver.make_rdm12s(self.civec, self.norb, self.nelec)
