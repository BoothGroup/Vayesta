from .solver import ClusterSolver, UClusterSolver
from .hamiltonian import is_uhf_ham, is_eb_ham

from vayesta.core.types import CISD_WaveFunction
from pyscf import ci
from ._uccsd_eris import uao2mo
import dataclasses


def CISD_Solver(hamil, *args, **kwargs):
    if is_eb_ham(hamil):
        raise NotImplementedError("Coupled electron-boson CISD solver not implemented.")
    if is_uhf_ham(hamil):
        return UCISD_Solver(hamil, *args, **kwargs)
    else:
        return RCISD_Solver(hamil, *args, **kwargs)


class RCISD_Solver(ClusterSolver):
    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        conv_tol: float = None  # Convergence tolerance. If None, use PySCF default

    def kernel(self, *args, **kwargs):
        mf_clus, frozen = self.hamil.to_pyscf_mf(allow_dummy_orbs=True)
        solver_class = self.get_solver_class()
        mycisd = solver_class(mf_clus, frozen=frozen)
        if self.opts.conv_tol:
            mycisd.conv_tol = self.opts.conv_tol
        ecisd, civec = mycisd.kernel()
        c0, c1, c2 = mycisd.cisdvec_to_amplitudes(civec)
        self.wf = CISD_WaveFunction(self.hamil.mo, c0, c1, c2)
        self.converged = True

    def get_solver_class(self):
        return ci.RCISD


class UCISD_Solver(UClusterSolver, RCISD_Solver):
    @dataclasses.dataclass
    class Options(RCISD_Solver.Options):
        pass

    def get_solver_class(self):
        return UCISD


class UCISD(ci.ucisd.UCISD):
    ao2mo = uao2mo