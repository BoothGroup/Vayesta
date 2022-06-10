import dataclasses
import copy
from timeit import default_timer as timer

import numpy as np

import pyscf
import pyscf.cc
import pyscf.cc.dfccsd
import pyscf.pbc
import pyscf.pbc.cc

from vayesta.core.util import *
from vayesta.core.types import Orbitals
from vayesta.core.types import WaveFunction
from vayesta.core.types import RCCSD_WaveFunction
from . import coupling
from .solver import ClusterSolver


class CCSD_Solver(ClusterSolver):

    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        # Convergence
        maxiter: int = 100              # Max number of iterations
        conv_tol: float = None          # Convergence energy tolerance
        conv_tol_normt: float = None    # Convergence amplitude tolerance
        t_as_lambda: bool = False       # If true, use Lambda=T approximation
        # Self-consistent mode
        sc_mode: int = None
        # DM
        dm_with_frozen: bool = False
        # Tailored-CCSD
        tcc: bool = False
        tcc_fci_opts: dict = dataclasses.field(default_factory=dict)
        # Active space methods
        c_cas_occ: np.array = None
        c_cas_vir: np.array = None
        # Lambda equations
        solve_lambda: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        solver_cls = self.get_solver_class()
        self.log.debugv("PySCF solver class= %r" % solver_cls)
        frozen = self.cluster.get_frozen_indices()
        # RCCSD does not support empty lists of frozen orbitals
        # For UCCSD len(frozen) is always 2, but empty lists are supported)
        if len(frozen) == 0:
            frozen = None
        mo_coeff = self.cluster.c_total
        solver = solver_cls(self.mf, mo_coeff=mo_coeff, mo_occ=self.mf.mo_occ, frozen=frozen)
        # Options
        if self.opts.maxiter is not None: solver.max_cycle = self.opts.maxiter
        if self.opts.conv_tol is not None: solver.conv_tol = self.opts.conv_tol
        if self.opts.conv_tol_normt is not None: solver.conv_tol_normt = self.opts.conv_tol_normt
        self.solver = solver
        self.eris = None

    def get_solver_class(self):
        # For 2D-systems the Coulomb repulsion is not PSD
        # Density-fitted CCSD does not support non-PSD three-center integrals,
        # thus we need a four-center formulation, where non PSD elements can be summed in
        if self.base.boundary_cond in ('periodic-1D', 'periodic-2D'):
            return pyscf.cc.ccsd.CCSD
        if hasattr(self.mf, 'with_df') and self.mf.with_df is not None:
            return pyscf.cc.dfccsd.RCCSD
        return pyscf.cc.ccsd.CCSD

    def reset(self):
        super().reset()
        self.eris = None

    @property
    @deprecated()
    def t1(self):
        return self.wf.t1

    @property
    @deprecated()
    def t2(self):
        return self.wf.t2

    @property
    @deprecated()
    def l1(self):
        return self.wf.l1

    @property
    @deprecated()
    def l2(self):
        return self.wf.l2

    @deprecated()
    def get_t1(self):
        return self.t1

    @deprecated()
    def get_t2(self):
        return self.t2

    @deprecated()
    def get_c1(self, intermed_norm=True):
        """C1 in intermediate normalization."""
        if not intermed_norm:
            raise ValueError()
        return self.t1

    @deprecated()
    def get_c2(self, intermed_norm=True):
        """C2 in intermediate normalization."""
        if not intermed_norm:
            raise ValueError()
        return self.t2 + einsum('ia,jb->ijab', self.t1, self.t1)

    @deprecated()
    def get_l1(self, t_as_lambda=None, solve_lambda=True):
        if t_as_lambda is None:
            t_as_lambda = self.opts.t_as_lambda
        if t_as_lambda:
            return self.t1
        if self.l1 is None:
            if not solve_lambda:
                return None
            self.solve_lambda()
        return self.l1

    @deprecated()
    def get_l2(self, t_as_lambda=None, solve_lambda=True):
        if t_as_lambda is None:
            t_as_lambda = self.opts.t_as_lambda
        if t_as_lambda:
            return self.t2
        if self.l2 is None:
            if not solve_lambda:
                return None
            self.solve_lambda()
        return self.l2

    def get_eris(self):
        self.log.debugv("Getting ERIs for type(self.solver)= %r", type(self.solver))
        with log_time(self.log.timing, "Time for 2e-integral transformation: %s"):
            self.eris = self.base.get_eris_object(self.solver)
        return self.eris

    def get_init_guess(self):
        return {'t1' : self.t1 , 't2' : self.t2}

    def kernel(self, t1=None, t2=None, eris=None, l1=None, l2=None, coupled_fragments=None, t_diagnostic=True):
        """

        Parameters
        ----------
        t1: array, optional
            Initial guess for T1 amplitudes. Default: None.
        t2: array, optional
            Initial guess for T2 amplitudes. Default: None.
        l1: array, optional
            Initial guess for L1 amplitudes. Default: None.
        l2: array, optional
            Initial guess for L2 amplitudes. Default: None.
        """

        if coupled_fragments is None:
            coupled_fragments = self.fragment.opts.coupled_fragments

        # Integral transformation
        if eris is None: eris = self.get_eris()

        # Add additional potential
        if self.v_ext is not None:
            self.log.debugv("Adding self.v_ext to eris.fock")
            # Make sure there are no side effects:
            eris = copy.copy(eris)
            # Replace fock instead of modifying it!
            if self.is_rhf:
                eris.fock = (eris.fock + self.v_ext)
            else:
                eris.focka = eris.fock[0] + self.v_ext[0]
                eris.fockb = eris.fock[1] + self.v_ext[1]
                eris.fock = (eris.focka, eris.fockb)

        # Tailored CC
        if self.opts.tcc:
            self.log.info("Adding tailor function to CCSD.")
            self.solver.callback = coupling.make_cas_tcc_function(
                    self, c_cas_occ=self.opts.c_cas_occ, c_cas_vir=self.opts.c_cas_vir, eris=eris)

        elif self.opts.sc_mode and self.base.iteration > 1:
            self.log.info("Adding tailor function to CCSD.")
            self.solver.callback = coupling.make_cross_fragment_tcc_function(self, mode=self.opts.sc_mode)

        # This should include the SC mode?
        elif coupled_fragments and np.all([x.results is not None for x in coupled_fragments]):
            self.log.info("Adding tailor function to CCSD.")
            self.solver.callback = coupling.make_cross_fragment_tcc_function(self, mode=(self.opts.sc_mode or 3),
                                                                                coupled_fragments=coupled_fragments)

        t0 = timer()
        self.log.info("Solving CCSD-equations %s initial guess...", "with" if (t2 is not None) else "without")
        self.solver.kernel(t1=t1, t2=t2, eris=eris)
        if not self.solver.converged:
            self.log.error("%s not converged!", self.__class__.__name__)
        else:
            self.log.debugv("%s converged.", self.__class__.__name__)
        self.converged = self.solver.converged
        self.e_corr = self.solver.e_corr
        if self.is_rhf:
            self.log.debugv("tr(T1)= %.8f", np.trace(self.solver.t1))
        else:
            self.log.debugv("tr(alpha-T1)= %.8f", np.trace(self.solver.t1[0]))
            self.log.debugv("tr( beta-T1)= %.8f", np.trace(self.solver.t1[1]))

        self.log.debug("Cluster: E(corr)= % 16.8f Ha", self.solver.e_corr)
        self.log.timing("Time for CCSD:  %s", time_string(timer()-t0))

        if hasattr(self.solver, '_norm_dt1'):
            self.log.debug("Tailored CC: |dT1|= %.2e |dT2|= %.2e", self.solver._norm_dt1, self.solver._norm_dt2)
            del self.solver._norm_dt1, self.solver._norm_dt2

        if self.opts.solve_lambda:
            #self.solve_lambda(eris=eris)
            self.log.info("Solving lambda-equations with%s initial guess...", ("out" if (l2 is None) else ""))
            self.solver.solve_lambda(l1=l1, l2=l2, eris=eris)
            if not self.solver.converged_lambda:
                self.log.error("Lambda-equations not converged!")

        if t_diagnostic: self.t_diagnostic()

        self.wf = WaveFunction.from_pyscf(self.solver)

    def t_diagnostic(self):
        self.log.info("T-Diagnostic")
        self.log.info("------------")
        try:
            dg_t1 = self.solver.get_t1_diagnostic()
            dg_d1 = self.solver.get_d1_diagnostic()
            dg_d2 = self.solver.get_d2_diagnostic()
            dg_t1_msg = "good" if dg_t1 <= 0.02 else "inadequate!"
            dg_d1_msg = "good" if dg_d1 <= 0.02 else ("fair" if dg_d1 <= 0.05 else "inadequate!")
            dg_d2_msg = "good" if dg_d2 <= 0.15 else ("fair" if dg_d2 <= 0.18 else "inadequate!")
            fmt = 3*"  %2s= %.3f (%s)"
            self.log.info(fmt, "T1", dg_t1, dg_t1_msg, "D1", dg_d1, dg_d1_msg, "D2", dg_d2, dg_d2_msg)
            # good: MP2~CCSD~CCSD(T) / fair: use MP2/CCSD with caution
            self.log.info("  (T1<0.02: good / D1<0.02: good, D1<0.05: fair / D2<0.15: good, D2<0.18: fair)")
            if dg_t1 > 0.02 or dg_d1 > 0.05 or dg_d2 > 0.18:
                self.log.info("  at least one diagnostic indicates that CCSD is not adequate!")
        except Exception as e:
            self.log.error("Exception in T-diagnostic: %s", e)

    def couple_iterations(self, fragments):
        self.solver.callback = coupling.couple_ccsd_iterations(self, fragments)

    def _debug_exact_wf(self, wf):
        mo = Orbitals(self.cluster.c_active, occ=self.cluster.nocc_active)
        # Project onto cluster:
        ovlp = self.fragment.base.get_ovlp()
        ro = dot(wf.mo.coeff_occ.T, ovlp, mo.coeff_occ)
        rv = dot(wf.mo.coeff_vir.T, ovlp, mo.coeff_vir)
        t1 = dot(ro.T, wf.t1, rv)
        t2 = einsum('Ii,Jj,IJAB,Aa,Bb->ijab', ro, ro, wf.t2, rv, rv)
        if wf.l1 is not None:
            l1 = dot(ro.T, wf.l1, rv)
            l2 = einsum('Ii,Jj,IJAB,Aa,Bb->ijab', ro, ro, wf.l2, rv, rv)
        else:
            l1 = l2 = None
        self.wf = RCCSD_WaveFunction(mo, t1, t2, l1=l1, l2=l2)
        self.converged = True

    def _debug_random_wf(self):
        mo = Orbitals(self.cluster.c_active, occ=self.cluster.nocc_active)
        t1 = np.random.rand(mo.nocc, mo.nvir)
        l1 = np.random.rand(mo.nocc, mo.nvir)
        t2 = np.random.rand(mo.nocc, mo.nocc, mo.nvir, mo.nvir)
        l2 = np.random.rand(mo.nocc, mo.nocc, mo.nvir, mo.nvir)
        self.wf = RCCSD_WaveFunction(mo, t1, t2, l1=l1, l2=l2)
        self.converged = True


class UCCSD_Solver(CCSD_Solver):

    def get_solver_class(self):
        # No DF-UCCSD class in PySCF
        # Molecular UCCSD does not support DF either!
        if self.base.boundary_cond.startswith('periodic'):
            return pyscf.pbc.cc.ccsd.UCCSD
        return pyscf.cc.uccsd.UCCSD

    @deprecated()
    def get_c2(self, intermed_norm=True):
        """C2 in intermediate normalization."""
        if not intermed_norm:
            raise ValueError()
        ta, tb = self.t1
        taa, tab, tbb = self.t2
        caa = taa + einsum('ia,jb->ijab', ta, ta) - einsum('ib,ja->ijab', ta, ta)
        cbb = tbb + einsum('ia,jb->ijab', tb, tb) - einsum('ib,ja->ijab', tb, tb)
        cab = tab + einsum('ia,jb->ijab', ta, tb)
        return (caa, cab, cbb)

    def t_diagnostic(self):
        """T diagnostic not implemented for UCCSD in PySCF."""
        self.log.info("T diagnostic not implemented for UCCSD in PySCF.")
