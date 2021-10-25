import dataclasses
import copy
from typing import Union
from timeit import default_timer as timer

import numpy as np

import pyscf
import pyscf.cc
import pyscf.cc.dfccsd

from vayesta.core.util import *
from vayesta.ewf import coupling
from .solver import ClusterSolver


class CCSD_Solver(ClusterSolver):

    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        # Convergence
        maxiter: int = 200              # Max number of iterations
        conv_tol: float = None          # Convergence energy tolerance
        conv_tol_normt: float = None    # Convergence amplitude tolerance
        # solve_lambda:
        # True:     Always solve lambda eq.
        # 'auto':   solve lambda eq. if DMs are requested via make_rdm1 or make_rdm2
        # False:    Never solve lambda eq. If DMs are requested, they are build with Lambda=T approximation
        solve_lambda: Union[bool, str] = 'auto'
        # Self-consistent mode
        #sc_mode: int = NotSet
        sc_mode: int = None
        # DM
        #dm_with_frozen: bool = NotSet
        dm_with_frozen: bool = False
        # EOM CCSD
        #eom_ccsd: list = NotSet  # {'IP', 'EA', 'EE-S', 'EE-D', 'EE-SF'}
        eom_ccsd: list = dataclasses.field(default_factory=list)
        #eom_ccsd_nroots: int = NotSet
        eom_ccsd_nroots: int = 3
        # Tailored-CCSD
        tcc: bool = False
        tcc_fci_opts: dict = dataclasses.field(default_factory=dict)
        # Active space methods
        c_cas_occ: np.array = None
        c_cas_vir: np.array = None

    @dataclasses.dataclass
    class Results(ClusterSolver.Results):
        t1: np.array = None
        t2: np.array = None
        l1: np.array = None
        l2: np.array = None
        solved_lambda: bool = False
        # EOM-CCSD
        ip_energy: np.array = None
        ip_coeff: np.array = None
        ea_energy: np.array = None
        ea_coeff: np.array = None
        # EE
        ee_s_energy: np.array = None
        ee_t_energy: np.array = None
        ee_sf_energy: np.array = None
        ee_s_coeff: np.array = None
        ee_t_coeff: np.array = None
        ee_sf_coeff: np.array = None

        def get_init_guess(self):
            """Get initial guess for another CCSD calculations from results."""
            return {'t1' : self.t1 , 't2' : self.t2, 'l1' : self.l1, 'l2' : self.l2}

    SOLVER_CLS = pyscf.cc.ccsd.CCSD
    SOLVER_CLS_DF = pyscf.cc.dfccsd.RCCSD

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # For 2D-systems the Coulomb repulsion is not PSD
        # Density-fitted CCSD does not support non-PSD three-center integrals,
        # thus we need a four-center formulation, where non PSD elements can be summed in
        if (self.base.boundary_cond not in ('periodic-1D', 'periodic-2D')
                and hasattr(self.mf, 'with_df') and self.mf.with_df is not None):
            cls = self.SOLVER_CLS_DF
        else:
            cls = self.SOLVER_CLS
        self.log.debug("CCSD class= %r" % cls)
        solver = cls(self.mf, mo_coeff=self.mo_coeff, mo_occ=self.mo_occ, frozen=self.get_frozen_indices())
        # Options
        if self.opts.maxiter is not None: solver.max_cycle = self.opts.maxiter
        if self.opts.conv_tol is not None: solver.conv_tol = self.opts.conv_tol
        if self.opts.conv_tol_normt is not None: solver.conv_tol_normt = self.opts.conv_tol_normt
        self.solver = solver

    def get_eris(self):
        with log_time(self.log.timing, "Time for AO->MO transformation: %s"):
            eris = self.base.get_eris_object(self.solver)
        return eris

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
        if self.opts.v_ext is not None:
            self.log.debugv("Adding self.opts.v_ext to eris.fock")
            # Make sure there are no side effects:
            eris = copy.copy(eris)
            # Replace fock instead of modifying it!
            eris.fock = (eris.fock + self.opts.v_ext)
        self.log.debugv("sum(eris.mo_energy)= %.8e", sum(eris.mo_energy))
        self.log.debugv("Tr(eris.fock)= %.8e", np.trace(eris.fock))

        # Tailored CC
        if self.opts.tcc:
            self.log.info("Adding tailor function to CCSD.")
            self.solver.tailor_func = coupling.make_cas_tcc_function(
                    self, c_cas_occ=self.opts.c_cas_occ, c_cas_vir=self.opts.c_cas_vir, eris=eris).__get__(self.solver)

        elif self.opts.sc_mode and self.base.iteration > 1:
            # __get__(self.solver) to bind the tailor function as a method, rather than just a callable attribute
            self.log.info("Adding tailor function to CCSD.")
            self.solver.tailor_func = coupling.make_cross_fragment_tcc_function(self, mode=self.opts.sc_mode).__get__(self.solver)

        # This should include the SC mode?
        elif coupled_fragments and np.all([x.results is not None for x in coupled_fragments]):
            self.log.info("Adding tailor function to CCSD.")
            self.solver.tailor_func = coupling.make_cross_fragment_tcc_function(self, mode=(self.opts.sc_mode or 3),
                    coupled_fragments=coupled_fragments).__get__(self.solver)

        t0 = timer()
        self.log.info("Solving CCSD-equations %s initial guess...", "with" if (t2 is not None) else "without")
        self.solver.kernel(t1=t1, t2=t2, eris=eris)
        if not self.solver.converged:
            self.log.error("CCSD not converged!")
        else:
            self.log.debugv("CCSD converged.")

        self.log.debug("Cluster: E(corr)= % 16.8f Ha", self.solver.e_corr)
        self.log.timing("Time for CCSD:  %s", time_string(timer()-t0))

        if hasattr(self.solver, '_norm_dt1'):
            self.log.debug("Tailored CC: |dT1|= %.2e |dT2|= %.2e", self.solver._norm_dt1, self.solver._norm_dt2)
            del self.solver._norm_dt1, self.solver._norm_dt2

        if t_diagnostic: self.t_diagnostic()

        results = self.Results(
                converged=self.solver.converged, e_corr=self.solver.e_corr, c_occ=self.c_active_occ, c_vir=self.c_active_vir,
                t1=self.solver.t1, t2=self.solver.t2)

        if self.opts.solve_lambda == 'auto':
            solve_lambda = (self.opts.make_rdm1 or self.opts.make_rdm2)
        else:
            solve_lambda = self.opts.solve_lambda

        if solve_lambda:
            t0 = timer()
            self.log.info("Solving lambda-equations %s initial guess...", "with" if (l2 is not None) else "without")
            l1, l2 = self.solver.solve_lambda(l1=l1, l2=l2, eris=eris)
            self.log.info("Lambda equations done. Lambda converged: %r", self.solver.converged_lambda)
            if self.solver.converged_lambda:
                results.solved_lambda = True
            else:
                self.log.error("Solution of lambda-equation not converged!")
            self.log.timing("Time for lambda-equations: %s", time_string(timer()-t0))
        # Use Lambda=T approximation
        else:
            l1 = self.solver.t1
            l2 = self.solver.t2
        results.l1, results.l2 = l1, l2

        if self.opts.make_rdm1:
            self.log.debug("Making RDM1...")
            results.dm1 = self.solver.make_rdm1(l1=l1, l2=l2, with_frozen=self.opts.dm_with_frozen)
        if self.opts.make_rdm2:
            self.log.debug("Making RDM2...")
            results.dm2 = self.solver.make_rdm2(l1=l1, l2=l2, with_frozen=self.opts.dm_with_frozen)

        if 'IP' in self.opts.eom_ccsd:
            results.ip_energy, results.ip_coeff = self.eom_ccsd('IP', eris)
        if 'EA' in self.opts.eom_ccsd:
            results.ea_energy, results.ea_coeff = self.eom_ccsd('EA', eris)
        if 'EE-S' in self.opts.eom_ccsd:
            results.ee_s_energy, results.ee_s_coeff = self.eom_ccsd('EE-S', eris)
        if 'EE-T' in self.opts.eom_ccsd:
            results.ee_t_energy, results.ee_t_coeff = self.eom_ccsd('EE-T', eris)
        if 'EE-SF' in self.opts.eom_ccsd:
            results.ee_sf_energy, results.ee_sf_coeff = self.eom_ccsd('EE-SF', eris)

        return results

    def t_diagnostic(self):
        self.log.info("T-Diagnostic")
        self.log.info("------------")
        try:
            dg_t1 = self.solver.get_t1_diagnostic()
            dg_d1 = self.solver.get_d1_diagnostic()
            dg_d2 = self.solver.get_d2_diagnostic()
            self.log.info("  (T1<0.02: good / D1<0.02: good, D1<0.05: fair / D2<0.15: good, D2<0.18: fair)")
            self.log.info("  (good: MP2~CCSD~CCSD(T) / fair: use MP2/CCSD with caution)")
            dg_t1_msg = "good" if dg_t1 <= 0.02 else "inadequate!"
            dg_d1_msg = "good" if dg_d1 <= 0.02 else ("fair" if dg_d1 <= 0.05 else "inadequate!")
            dg_d2_msg = "good" if dg_d2 <= 0.15 else ("fair" if dg_d2 <= 0.18 else "inadequate!")
            fmtstr = "  > %2s= %6g (%s)"
            self.log.info(fmtstr, "T1", dg_t1, dg_t1_msg)
            self.log.info(fmtstr, "D1", dg_d1, dg_d1_msg)
            self.log.info(fmtstr, "D2", dg_d2, dg_d2_msg)
            if dg_t1 > 0.02 or dg_d1 > 0.05 or dg_d2 > 0.18:
                self.log.warning("  some diagnostic(s) indicate CCSD may not be adequate.")
        except Exception as e:
            self.log.error("Exception in T-diagnostic: %s", e)

    def eom_ccsd(self, kind, eris, nroots=None):
        nroots = nroots or self.opts.eom_ccsd_nroots
        kind = kind.upper()
        assert kind in ('IP', 'EA', 'EE-S', 'EE-T', 'EE-SF')
        self.log.info("Running %s-EOM-CCSD (nroots=%d)...", kind, nroots)
        cc = self.solver
        eom_funcs = {
                'IP' : cc.ipccsd , 'EA' : cc.eaccsd,
                'EE-S' : cc.eomee_ccsd_singlet,
                'EE-T' : cc.eomee_ccsd_triplet,
                'EE-SF' : cc.eomsf_ccsd,}
        t0 = timer()
        e, c = eom_funcs[kind](nroots=nroots, eris=eris)
        self.log.timing("Time for %s-EOM-CCSD:  %s", kind, time_string(timer()-t0))
        if nroots == 1:
            e, c = np.asarray([e]), np.asarray([c])
        fmt = "%s-EOM-CCSD energies:" + len(e) * "  %+14.8f"
        self.log.info(fmt, kind, *e)
        return e, c
