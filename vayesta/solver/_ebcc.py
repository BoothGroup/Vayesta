import dataclasses
from types import SimpleNamespace

import numpy as np

from pyscf import lib

from vayesta.solver.solver import EBClusterSolver
from vayesta.core.types.wf import REBCC_WaveFunction, UEBCC_WaveFunction


class REBCC_Solver(EBClusterSolver):

    WaveFunction = REBCC_WaveFunction

    @dataclasses.dataclass
    class Options(EBClusterSolver.Options):
        # Ansatz
        fermion_excitations: str = "SD"
        boson_excitations: str = ""
        fermion_coupling_rank: int = 0
        boson_coupling_rank: int = 0

        # Convergence
        maxiter: int = 100
        conv_tol: float = None
        conv_tol_normt: float = None
        diis_space: int = None

        # Lambda equations
        solve_lambda: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        solver_cls = self.get_solver()
        self.log.debugv("ebcc solver class= %r" % solver_cls)

        mo_coeff = self.cluster.c_active
        mo_occ = self.get_occ()

        solver = solver_cls(
                self.mf,
                log=self.log,
                fermion_excitations=self.opts.fermion_excitations,
                boson_excitations=self.opts.boson_excitations,
                fermion_coupling_rank=self.opts.fermion_coupling_rank,
                boson_coupling_rank=self.opts.boson_coupling_rank,
                mo_coeff=mo_coeff,
                mo_occ=mo_occ,
                g=self.fragment.couplings,
                omega=self.fragment.bos_freqs,
                shift=self.opts.polaritonic_shift,
        )

        if self.opts.maxiter is not None: solver.options.max_iter = self.opts.maxiter
        if self.opts.conv_tol is not None: solver.options.e_tol = self.opts.conv_tol
        if self.opts.conv_tol_normt is not None: solver.options.t_tol = self.opts.conv_tol_normt
        if self.opts.diis_space is not None: solver.options.diis_space = self.opts.diis_space

        self.solver = solver
        self.fock = None
        self.eris = None

    def get_solver(self):
        from ebcc import REBCC
        return REBCC

    def get_wavefunction(self):
        return REBCC_WaveFunction(self.solver)

    def reset(self):
        super().reset()
        self.fock = None
        self.eris = None

    def get_occ(self):
        mo_occ = np.concatenate([2] * self.cluster.nocc_active + [0] * self.cluster.nvir_active)
        return mo_occ

    def get_eris(self):
        self.log.debugv("Getting ERIs for type(self.solver)= %r", type(self.solver))
        with log_time(self.log.timing, "Time for 2e-integral transformation: %s"):
            array = self.get_eris_array(self.solver.mo_coeff, compact=False)
            self.eris = self.solver.get_eris(eris=array)
        return self.eris

    def add_screening(self, eris, seris_ov):
        raise NotImplementedError

    def add_potential(self, eris, v_ext):
        raise NotImplementedError

    def kernel(self, amplitudes=None, lambdas=None, fock=None, eris=None, coupled_fragments=None, seris_ov=None):
        """Run the solver.
        """

        if coupled_fragments is None:
            coupled_fragments = self.fragment.opts.coupled_fragments

        if eris is None:
            eris = self.eris
        if eris is None:
            eris = self.eris()

        # Add screening [optional]
        if seris_ov is not None:
            eris = self.add_screening(eris, seris_ov)

        # Add additional potential [optional]
        if self.v_ext is not None:
            eris = self.add_potential(eris, self.v_ext)

        self.log.info(
                "Solving CCSD-equations with%s initial guess...",
                "out" if amplitudes is None else "",
        )
        with log_time(self.log.info, "Time for T-equations: %s"):
            self.solver.kernel(amplitudes=amplitudes, lambdas=lambdas)
        if not self.solver.converged:
            self.log.error("%s not converged!", self.__class__.__name__)
        else:
            self.log.debugv("%s converged.", self.__class__.__name__)

        self.converged = self.solver.converged
        self.e_corr = self.solver.e_corr

        if self.is_rhf:
            self.log.debugv("tr(T1)= %.8f", np.trace(self.solver.t1))
        else:
            self.log.debugv("tr(alpha-T1)= %.8f", np.trace(self.solver.t1.aa))
            self.log.debugv("tr( beta-T1)= %.8f", np.trace(self.solver.t1.bb))
        self.log.debug("Cluster: E(corr)= % 16.f Ha", self.solver.e_corr)

        if self.opts.solve_lambda:
            self.log.info(
                    "Solving lambda-equations with%s initial guess..",
                    "out" if lambdas is None else "",
            )
            if not self.solver.converged_lambda:
                self.log.error("Lambda-equations not converged!")

        # Remove screening (for energy calculation etc)
        if hasattr(eris, "restore_bare"):
            eris.restore_bare()

        mo = Orbitals(self.cluster.c_active, occ=self.cluster.nocc_active)
        self.wf = self.get_wavefunction()


class UEBCC_Solver(REBCC_Solver):

    def get_solver(self):
        from ebcc import UEBCC
        return UEBCC

    def get_wavefunction(self):
        return UEBCC_WaveFunction(self.solver)

    def get_occ(self):
        mo_occ_a = np.concatenate([1] * self.cluster.nocc_active[0] + [0] * self.cluster.nvir_active[0])
        mo_occ_b = np.concatenate([1] * self.cluster.nocc_active[1] + [0] * self.cluster.nvir_active[1])
        return (mo_occ_a, mo_occ_b)
