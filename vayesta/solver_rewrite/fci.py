import dataclasses
from .solver import ClusterSolver, UClusterSolver
from vayesta.core.types import Orbitals
from vayesta.core.types import WaveFunction, FCI_WaveFunction
import pyscf.fci


class FCI_Solver(ClusterSolver):
    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        threads: int = 1            # Number of threads for multi-threaded FCI
        max_cycle: int = 300
        lindep: float = None        # Linear dependency tolerance. If None, use PySCF default
        conv_tol: float = None      # Convergence tolerance. If None, use PySCF default
        solver_spin: bool = True    # Use direct_spin1 if True, or direct_spin0 otherwise
        fix_spin: float = 0.0       # If set to a number, the given S^2 value will be enforced
        #fix_spin_penalty: float = 1.0
        fix_spin_penalty: float = 1e3

    def kernel_solver(self, mf_clus, eris_energy=None):
        # Pyscf can detect restricted or not from mean-field..
        solver = pyscf.fci.FCI(mf_clus, singlet=not self.opts.solver_spin)
        solver.conv_tol = self.opts.conv_tol
        e, civec = solver.kernel()
        self.converged = solver.converged
        mo = Orbitals(self.cluster.c_active, occ=self.cluster.nocc_active)
        self.wf = FCI_WaveFunction(mo, civec)


class UFCI_Solver(UClusterSolver, FCI_Solver):

    @dataclasses.dataclass
    class Options(FCI_Solver.Options):
        fix_spin: float = None
