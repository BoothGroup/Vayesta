from .solver import EBClusterSolver, UEBClusterSolver
from .eb_fci import EBFCI
from vayesta.core.types import Orbitals, WaveFunction
import dataclasses

class EB_EBFCI_Solver(EBClusterSolver):

    @dataclasses.dataclass
    class Options(EBClusterSolver.Options):
        polaritonic_shift: bool = True

    def kernel_solver(self, mf_clus, freqs, couplings):

        solver = EBFCI(mf_clus, freqs, couplings)
        e_fci, civec = solver.kernel()
        mo = Orbitals(self.cluster.c_active, occ=self.cluster.nocc_active)
        self.wf = WaveFunction(mo)



