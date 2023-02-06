from .solver import ClusterSolver, UClusterSolver
from .eb_fci import REBFCI, UEBFCI
from vayesta.core.types import WaveFunction
import dataclasses
import numpy as np


class EB_EBFCI_Solver(ClusterSolver):
    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        # Convergence
        max_cycle: int = 100  # Max number of iterations
        conv_tol: float = None  # Convergence energy tolerance
        # Parameterisation
        max_boson_occ: int = 2

    def get_solver(self, *args, **kwargs):
        return REBFCI(*args, **kwargs)

    def kernel(self):
        solver = self.get_solver(self.hamil, self.hamil.bos_freqs, self.hamil.couplings,
                                 max_boson_occ=self.opts.max_boson_occ, conv_tol=self.opts.conv_tol or 1e-12)
        e_fci, civec = solver.kernel()
        self.wf = WaveFunction(self.hamil.mo)
        self.wf.make_rdm1 = lambda *args, **kwargs: solver.make_rdm1(*args, **kwargs)
        self.wf.make_rdm2 = lambda *args, **kwargs: solver.make_rdm2(*args, **kwargs)
        self.wf.make_rdmeb = lambda *args, **kwargs: np.array(solver.make_rdm_eb(*args, **kwargs)) + \
                                                     np.array(
                                                         self.hamil.get_eb_dm_polaritonic_shift(self.wf.make_rdm1()))
        self.wf.make_dd_moms = lambda max_mom, *args, **kwargs: solver.make_dd_moms(max_mom, *args, **kwargs)


class EB_UEBFCI_Solver(UClusterSolver, EB_EBFCI_Solver):

    def get_solver(self, *args, **kwargs):
        return UEBFCI(*args, **kwargs)
