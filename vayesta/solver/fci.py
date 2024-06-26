import dataclasses

import numpy as np
import pyscf.fci
import pyscf.fci.addons

from vayesta.core.types import FCI_WaveFunction
from vayesta.core.types.wf.fci import UFCI_WaveFunction_w_dummy
from vayesta.core.util import log_time
from vayesta.solver.cisd import RCISD_Solver, UCISD_Solver
from vayesta.solver.solver import ClusterSolver, UClusterSolver


class FCI_Solver(ClusterSolver):
    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        threads: int = 1  # Number of threads for multi-threaded FCI
        max_cycle: int = 300
        lindep: float = None  # Linear dependency tolerance. If None, use PySCF default
        conv_tol: float = 1e-12  # Convergence tolerance. If None, use PySCF default
        solver_spin: bool = True  # Use direct_spin1 if True, or direct_spin0 otherwise
        fix_spin: float = 0.0  # If set to a number, the given S^2 value will be enforced
        fix_spin_penalty: float = 1.0  # Penalty for fixing spin
        davidson_only: bool = True
        init_guess: str = "default"
        init_guess_noise: float = 1e-5
        n_moments: tuple = None

    cisd_solver = RCISD_Solver

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        solver_cls = self.get_solver_class()
        # This just uses mol to initialise various outputting defaults.
        solver = solver_cls(self.hamil.orig_mf.mol)
        self.log.debugv("type(solver)= %r", type(solver))
        # Set options
        if self.opts.init_guess == "default":
            self.opts.init_guess = "CISD"
        if self.opts.threads is not None:
            solver.threads = self.opts.threads
        if self.opts.conv_tol is not None:
            solver.conv_tol = self.opts.conv_tol
        if self.opts.lindep is not None:
            solver.lindep = self.opts.lindep
        if self.opts.max_cycle is not None:
            solver.max_cycle = self.opts.max_cycle
        if self.opts.davidson_only is not None:
            solver.davidson_only = self.opts.davidson_only
        if self.opts.fix_spin not in (None, False):
            spin = self.opts.fix_spin
            self.log.debugv("Fixing spin of FCI solver to S^2= %f", spin)
            solver = pyscf.fci.addons.fix_spin_(solver, shift=self.opts.fix_spin_penalty, ss=spin)
        self.solver = solver

    def get_solver_class(self):
        if self.opts.solver_spin:
            return pyscf.fci.direct_spin1.FCISolver
        return pyscf.fci.direct_spin0.FCISolver

    def kernel(self, ci=None):
        self.hamil.assert_equal_spin_channels()

        if self.opts.init_guess == "CISD":
            cisd = self.cisd_solver(self.hamil)
            cisd.kernel()
            ci = cisd.wf.as_fci().ci
        elif self.opts.init_guess in ["mf", "meanfield"]:
            ci = None

        if self.opts.init_guess_noise and ci is not None:
            ci += self.opts.init_guess_noise * np.random.random(ci.shape)

        heff, eris = self.hamil.get_integrals(with_vext=True)

        with log_time(self.log.timing, "Time for FCI: %s"):
            e_fci, self.civec = self.solver.kernel(heff, eris, self.hamil.ncas[0], self.hamil.nelec, ci0=ci)
        self.converged = self.solver.converged
        self.wf = FCI_WaveFunction(self.hamil.mo, self.civec)

        # Cluster spectral moments
        nmom = self.opts.n_moments
        if nmom is not None:
            try:
                from dyson.expressions import FCI
            except ImportError:
                self.log.error("Dyson not found - required for moment calculations")
                self.log.info("Skipping cluster moment calculations")
                return
            
            self.log.info("Calculating cluster FCI spectral moments %s"%str(nmom))
            mf_clus, frozen = self.hamil.to_pyscf_mf(allow_dummy_orbs=True, allow_df=True)

            with log_time(self.log.timing, "Time for hole moments: %s"):
                expr = FCI["1h"](mf_clus, e_ci=e_fci, c_ci=self.civec, h1e=heff, h2e=eris)
                self.hole_moments = expr.build_gf_moments(nmom[0])
            with log_time(self.log.timing, "Time for hole moments: %s"):    
                expr = FCI["1p"](mf_clus, e_ci=e_fci, c_ci=self.civec, h1e=heff, h2e=eris)
                self.particle_moments = expr.build_gf_moments(nmom[1])

class UFCI_Solver(UClusterSolver, FCI_Solver):
    @dataclasses.dataclass
    class Options(FCI_Solver.Options):
        fix_spin: float = None

    cisd_solver = UCISD_Solver

    def get_solver_class(self):
        return pyscf.fci.direct_uhf.FCISolver

    def kernel(self, ci=None):
        na, nb = self.hamil.ncas
        if na == nb:
            super().kernel(ci)
            return
        # Only consider this functionality when it's needed.
        if ci is None and self.opts.init_guess == "CISD":
            self.log.warning(
                "CISD initial guess not implemented for UHF FCI solver with different numbers of alpha and beta orbitals."
                "Using meanfield guess."
            )
        # Get dummy meanfield object, with padding, to represent the cluster.
        mf, orbs_to_freeze = self.hamil.to_pyscf_mf(allow_dummy_orbs=True, allow_df=False)
        # Get padded integrals from this.
        heff = mf.get_hcore()
        eris = mf._eri
        # Run calculation with them.
        with log_time(self.log.timing, "Time for FCI: %s"):
            e_fci, self.civec = self.solver.kernel(heff, eris, mf.mol.nao, self.hamil.nelec, ci0=ci)
        self.converged = self.solver.converged
        # Generate wavefunction object with dummy orbitals.
        self.wf = UFCI_WaveFunction_w_dummy(self.hamil.mo, self.civec, dummy_orbs=orbs_to_freeze)
