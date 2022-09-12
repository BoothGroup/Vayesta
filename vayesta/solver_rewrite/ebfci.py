from .solver import EBClusterSolver, UEBClusterSolver
from vayesta.core.types import Orbitals
from vayesta.solver_rewrite.eb_fci import ebfci_slow, uebfci_slow
class EB_EBFCI_Solver(EBClusterSolver):

    def kernel_solver(self, mf_clus, eris_energy=None):

        pass