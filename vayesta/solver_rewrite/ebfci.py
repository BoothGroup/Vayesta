from .solver import ClusterSolver, UClusterSolver
from .eb_fci import REBFCI, UEBFCI
from vayesta.core.types import Orbitals, WaveFunction
import dataclasses

class EB_EBFCI_Solver(ClusterSolver):

    def get_solver(self, *args, **kwargs):
        return REBFCI(*args, **kwargs)

    def kernel(self):
        solver = self.get_solver(self.hamil, self.hamil.bos_freqs, self.hamil.couplings)
        e_fci, civec = solver.kernel()
        self.wf = WaveFunction(self.hamil.mo)
        self.wf.make_rdm1 = lambda *args, **kwargs: solver.make_rdm1(*args, **kwargs)
        self.wf.make_rdm2 = lambda *args, **kwargs: solver.make_rdm2(*args, **kwargs)
        self.wf.make_rdmeb = lambda *args, **kwargs: solver.make_rdm_eb(*args, **kwargs)
        self.wf.make_dd_moms = lambda max_mom, *args, **kwargs: solver.make_dd_moms(max_mom, *args, **kwargs)

class EB_UEBFCI_Solver(UClusterSolver, EB_EBFCI_Solver):

    def get_solver(self, *args, **kwargs):
        return UEBFCI(*args, **kwargs)
