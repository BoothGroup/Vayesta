from .solver import ClusterSolver, UClusterSolver
from vayesta.core.types import CISD_WaveFunction
from pyscf import ci
from ._uccsd_eris import uao2mo


class RCISD_Solver(ClusterSolver):

    def kernel(self, *args, **kwargs):
        mf_clus, frozen = self.hamil.to_pyscf_mf(allow_dummy_orbs=True)
        solver_class = self.get_solver_class()
        mycisd = solver_class(mf_clus, frozen=frozen)
        ecisd, civec = mycisd.kernel()
        c0, c1, c2 = mycisd.cisdvec_to_amplitudes(civec)
        self.wf = CISD_WaveFunction(self.hamil.mo, c0, c1, c2)
        self.converged = True

    def get_solver_class(self):
        return ci.RCISD


class UCISD_Solver(UClusterSolver, RCISD_Solver):
    def get_solver_class(self):
        return UCISD


class UCISD(ci.ucisd.UCISD):
    ao2mo = uao2mo