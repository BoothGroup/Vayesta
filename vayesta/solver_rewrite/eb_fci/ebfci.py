from . import ebfci_slow, uebfci_slow
from pyscf import ao2mo
import numpy as np


class REBFCI:
    """Performs FCI on coupled electron-boson systems.
    Input:
        -system specification.
    Output:
        -FCI RDMs.
        -FCI response specification.
    """
    solver = ebfci_slow

    def __init__(self, mf, freqs, couplings, max_boson_occ=5, conv_tol=1e-12):
        self.mf = mf
        self.max_boson_occ = max_boson_occ
        self.freqs = freqs
        self.couplings = couplings
        self.conv_tol = conv_tol

    @property
    def norb(self):
        return self.mf.mol.nao

    @property
    def nelec(self):
        return self.mf.nelec[0]

    @property
    def nbos(self):
        return len(self.freqs.shape)

    def kernel(self, eris=None):
        # Get MO eris.
        h1e, eris = self.get_hamil(eris)
        ne = self.mf.mol.nelectron
        nc = self.mf.mol.nao
        self.e_fci, self.civec = self.solver.kernel(h1e, eris, self.couplings,
                                                    np.diag(self.freqs), self.norb, self.nelec, self.nbos,
                                                    self.max_boson_occ, tol=self.conv_tol)

        return self.e_fci, self.civec

    def get_hamil(self, eris=None):
        mo_coeff = self.mf.mo_coeff
        if eris is None:
            eris = ao2mo.full(self.mf._eri, mo_coeff, compact=False)
        h1e = np.dot(mo_coeff.T, np.dot(self.mf.get_hcore(), mo_coeff))
        return h1e, eris

    def make_rdm1(self):
        return self.solver.make_rdm1(self.civec, self.norb, self.nelec)

    def make_rdm2(self):
        dm1, dm2 = self.solver.make_rdm12(self.civec, self.norb, self.nelec)
        return dm2

    def make_rdm12(self):
        return self.solver.make_rdm12s(self.civec, self.norb, self.nelec)

    def make_rdm_eb(self):
        # Note this is always spin-resolved, since bosonic couplings can have spin-dependence.
        return self.solver.make_eb_rdm(self.civec, self.norb, self.nelec, self.nbos, self.max_boson_occ)

    def make_dd_moms(self, max_mom, dm1=None, coeffs=None, civec=None, eris=None):
        if civec is None:
            civec = self.civec
        h1e, eris = self.get_hamil(eris)

        if dm1 is None:
            dm1 = self.make_rdm1(civec)

        self.dd_moms = self.solver.calc_dd_resp_mom(
            civec, self.e_fci, max_mom, self.norb, self.nelec, self.nbos, h1e, eris,
            np.diag(self.freqs), self.couplings, self.max_boson_occ, dm1,
            coeffs=coeffs)
        return self.dd_moms


class UEBFCI(REBFCI):
    solver = uebfci_slow

    @property
    def nelec(self):
        return self.mf.nelec
