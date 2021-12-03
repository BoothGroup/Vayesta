import dataclasses

import numpy as np

import pyscf
import pyscf.ao2mo
import pyscf.ci
import pyscf.cc
import pyscf.mcscf
import pyscf.fci
import pyscf.fci.addons

from vayesta.core.util import *
from .solver2 import ClusterSolver
from vayesta.solver.fci2 import FCI_Solver, UFCI_Solver
from .eb_fci import ebfci_slow


class EBFCI_Solver(FCI_Solver):
    @dataclasses.dataclass
    class Options(FCI_Solver.Options):
        bos_occ_cutoff: int = NotSet

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

    def reset(self):
        super().reset()

    def kernel(self, ci0=None, eris=None):
        """Run FCI kernel."""

        if eris is None:
            eris = self.get_eris()
        heff = self.get_heff(eris)

        t0 = timer()
        e_fci, civec = ebfci_slow.kernel(heff, eris, self.fragment.couplings, np.diag(self.fragment.bos_freqs),
                                         self.ncas, self.nelec, self.nbos, self.opts.bos_occ_cutoff)

        if not self.solver.converged:
            self.log.error("EBFCI not converged!")
        else:
            self.log.debugv("EBFCI converged.")
        self.log.timing("Time for EBFCI: %s", time_string(timer() - t0))
        self.log.debugv("E(CAS)= %s", energy_string(e_fci))
        # TODO: This requires the E_core energy (and nuc-nuc repulsion)
        self.e_corr = np.nan
        self.converged = self.solver.converged
        s2, mult = self.solver.spin_square(self.civec, self.ncas, self.nelec)
        self.log.info("FCI: S^2= %.10f  multiplicity= %.10f", s2, mult)
        self.c0, self.c1, self.c2 = self.get_cisd_amps(self.civec)

    def make_rdm1(self, civec=None):
        if civec is None: civec = self.civec
        self.dm1 = self.solver.make_rdm1(civec, self.ncas, self.nelec)
        return self.dm1

    def make_rdm12(self, civec=None):
        if civec is None:
            civec = self.civec
        self.dm1, self.dm2 = ebfci_slow.make_rdm12e(civec, self.ncas, self.nelec)
        return self.dm1, self.dm2

    def make_rdm2(self, civec=None):
        if civec is None:
            civec = self.civec
        self.dm1, self.dm2 = ebfci_slow.make_rdm12e(civec, self.ncas, self.nelec)
        return self.make_rdm12(civec=civec)[1]
