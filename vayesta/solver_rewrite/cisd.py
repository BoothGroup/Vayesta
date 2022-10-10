from .solver import ClusterSolver, UClusterSolver
from vayesta.core.types import CISD_WaveFunction
from pyscf import ci
from ._uccsd_eris import uao2mo


class RCISD_Solver(ClusterSolver):

    def kernel(self, *args, **kwargs):
        mf_clus = self.hamil.to_pyscf_mf()
        solver_class = self.get_solver_class()
        mycisd = solver_class(mf_clus)
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