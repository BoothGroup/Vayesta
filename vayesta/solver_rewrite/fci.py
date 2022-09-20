import dataclasses
from .solver import ClusterSolver, UClusterSolver
from vayesta.core.types import Orbitals
from vayesta.core.types import WaveFunction, FCI_WaveFunction
from vayesta.core.util import log_time

import pyscf.fci
import pyscf.fci.addons


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        solver_cls = self.get_solver_class()
        # This just uses mol to initialise various outputting defaults.
        solver = solver_cls(self.hamil.mf.mol)
        self.log.debugv("type(solver)= %r", type(solver))
        # Set options
        if self.opts.threads is not None: solver.threads = self.opts.threads
        if self.opts.conv_tol is not None: solver.conv_tol = self.opts.conv_tol
        if self.opts.lindep is not None: solver.lindep = self.opts.lindep
        if self.opts.max_cycle is not None: solver.max_cycle = self.opts.max_cycle
        if self.opts.fix_spin not in (None, False):
            spin = self.opts.fix_spin
            self.log.debugv("Fixing spin of FCI solver to S^2= %f", spin)
            solver = pyscf.fci.addons.fix_spin_(solver, shift=self.opts.fix_spin_penalty, ss=spin)
        self.solver = solver

    def get_solver_class(self):
        if self.opts.solver_spin:
            return pyscf.fci.direct_spin1.FCISolver
        return pyscf.fci.direct_spin0.FCISolver

    def kernel(self):
        self.hamil.assert_equal_spin_channels()

        heff, eris = self.hamil.get_integrals(with_vext=True)

        with log_time(self.log.timing, "Time for FCI: %s"):
            e_fci, self.civec = self.solver.kernel(heff, eris, self.hamil.ncas[0], self.hamil.nelec)
        self.converged = self.solver.converged
        self.wf = FCI_WaveFunction(self.hamil.mo, self.civec)


class UFCI_Solver(UClusterSolver, FCI_Solver):

    @dataclasses.dataclass
    class Options(FCI_Solver.Options):
        fix_spin: float = None

    def get_solver_class(self):
        return pyscf.fci.direct_uhf.FCISolver