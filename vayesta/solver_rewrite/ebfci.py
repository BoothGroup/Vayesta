from .solver import ClusterSolver, UClusterSolver
from .eb_fci import EBFCI
from vayesta.core.types import Orbitals, WaveFunction
import dataclasses

class EB_EBFCI_Solver(ClusterSolver):

    def kernel(self):
        mf_clus = self.hamil.to_pyscf_mf()
        # This will auto-detect RHF vs UHF.
        solver = EBFCI(mf_clus, self.hamil.bos_freqs, self.hamil.couplings)
        e_fci, civec = solver.kernel()
        self.wf = WaveFunction(self.hamil.mo)
        raise NotImplementedError("Still need to properly plumb in EBFCI rdms.")

class EB_UEBFCI_Solver(UClusterSolver, EB_EBFCI_Solver):
    pass