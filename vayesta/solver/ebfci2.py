import dataclasses

import numpy as np

from vayesta.core.util import *
from .solver2 import ClusterSolver
from vayesta.solver.fci2 import FCI_Solver, UFCI_Solver
from .eb_fci import ebfci_slow, uebfci_slow


class EBFCI_Solver(FCI_Solver):
    @dataclasses.dataclass
    class Options(FCI_Solver.Options):
        max_boson_occ: int = NotSet

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_solver_class(self):
        # [TODO]- reinstate object oriented interface for EBFCI code.
        class DummySolver:
            def __init__(self, *args):
                pass

        return DummySolver

    @property
    def nbos(self):
        return self.fragment.nbos

    def kernel(self, ci0=None, eris=None):
        """Run FCI kernel."""

        if eris is None:
            eris = self.get_eris()
        heff = self.get_heff(eris)

        t0 = timer()
        self.e_fci, self.civec = ebfci_slow.kernel(heff, eris, self.fragment.couplings,
                                                   np.diag(self.fragment.bos_freqs), self.ncas, self.nelec, self.nbos,
                                                   self.opts.max_boson_occ)
        # Getting convergence detail out pretty complicated, and nonconvergence rare- just assume for now.
        self.log.timing("Time for EBFCI: %s", time_string(timer() - t0))
        self.log.debugv("E(CAS)= %s", energy_string(self.e_fci))
        # TODO: This requires the E_core energy (and nuc-nuc repulsion)
        self.e_corr = np.nan
        self.converged = True
        # s2, mult = self.solver.spin_square(self.civec, self.ncas, self.nelec)
        # self.log.info("FCI: S^2= %.10f  multiplicity= %.10f", s2, mult)
        # self.c0, self.c1, self.c2 = self.get_cisd_amps(self.civec)

    def make_rdm1(self, civec=None):
        if civec is None:
            civec = self.civec
        self.dm1 = ebfci_slow.make_rdm1(civec, self.ncas, self.nelec)
        return self.dm1

    def make_rdm12(self, civec=None):
        if civec is None:
            civec = self.civec
        self.dm1, self.dm2 = ebfci_slow.make_rdm12(civec, self.ncas, self.nelec)
        return self.dm1, self.dm2

    def make_rdm2(self, civec=None):
        if civec is None:
            civec = self.civec
        self.dm1, self.dm2 = ebfci_slow.make_rdm12(civec, self.ncas, self.nelec)
        return self.dm2

    def make_dd_moms(self, max_mom, coeffs, civec=None, eris=None):
        if civec is None:
            civec = self.civec
        if eris is None:
            eris = self.get_eris()
        heff = self.get_heff(eris)

        try:
            dm1 = self.dm1
        except AttributeError:
            dm1 = self.make_rdm1(civec)

        self.dd_moms = ebfci_slow.calc_dd_resp_mom(
            civec, self.e_fci, max_mom, self.ncas, self.nelec, self.nbos, heff, eris,
            np.diag(self.fragment.bos_freqs), self.fragment.couplings, self.opts.max_boson_occ, dm1,
            coeffs=coeffs)
        return self.dd_moms

    def make_rdm_eb(self, civec=None):
        # Note this is always spin-resolved, since bosonic couplings can have spin-dependence.
        if civec is None:
            civec = self.civec
        self.dm_eb = ebfci_slow.make_eb_rdm(civec, self.ncas, self.nelec, self.nbos, self.opts.max_boson_occ)
        return self.dm_eb


class UEBFCI_Solver(EBFCI_Solver, UFCI_Solver):

    def kernel(self, ci0=None, eris=None):
        """Run FCI kernel."""

        if eris is None:
            eris = self.get_eris()
        heff = self.get_heff(eris)

        t0 = timer()
        self.e_fci, self.civec = uebfci_slow.kernel(heff, eris, self.fragment.couplings,
                                                   np.diag(self.fragment.bos_freqs), self.ncas, self.nelec, self.nbos,
                                                   self.opts.max_boson_occ)
        # Getting convergence detail out pretty complicated, and nonconvergence rare- just assume for now.
        self.log.timing("Time for EBFCI: %s", time_string(timer() - t0))
        self.log.debugv("E(CAS)= %s", energy_string(self.e_fci))
        # TODO: This requires the E_core energy (and nuc-nuc repulsion)
        self.e_corr = np.nan
        self.converged = True
        # s2, mult = self.solver.spin_square(self.civec, self.ncas, self.nelec)
        # self.log.info("FCI: S^2= %.10f  multiplicity= %.10f", s2, mult)
        # self.c0, self.c1, self.c2 = self.get_cisd_amps(self.civec)

    def make_rdm1(self, civec=None):
        if civec is None:
            civec = self.civec
        self.dm1 = uebfci_slow.make_rdm1(civec, self.ncas, self.nelec)
        return self.dm1

    def make_rdm12(self, civec=None):
        if civec is None:
            civec = self.civec
        self.dm1, self.dm2 = uebfci_slow.make_rdm12s(civec, self.ncas, self.nelec)
        return self.dm1, self.dm2

    def make_rdm2(self, civec=None):
        if civec is None:
            civec = self.civec
        self.dm1, self.dm2 = uebfci_slow.make_rdm12(civec, self.ncas, self.nelec)
        return self.dm2

    def make_dd_moms(self, max_mom, coeffs, civec=None, eris=None):
        if civec is None:
            civec = self.civec
        if eris is None:
            eris = self.get_eris()
        heff = self.get_heff(eris)

        try:
            dm1 = self.dm1
        except AttributeError:
            dm1 = self.make_rdm1(civec)

        self.dd_moms = uebfci_slow.calc_dd_resp_mom(
            civec, self.e_fci, max_mom, self.ncas, self.nelec, self.nbos, heff, eris,
            np.diag(self.fragment.bos_freqs), self.fragment.couplings, self.opts.max_boson_occ, dm1,
            coeffs=coeffs)
        return self.dd_moms

    def make_rdm_eb(self, civec=None):
        # Note this is always spin-resolved, since bosonic couplings can have spin-dependence.
        if civec is None:
            civec = self.civec
        self.dm_eb = uebfci_slow.make_eb_rdm(civec, self.ncas, self.nelec, self.nbos, self.opts.max_boson_occ)
        return self.dm_eb
