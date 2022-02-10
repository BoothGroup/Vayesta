import dataclasses
import copy
from typing import Union
from timeit import default_timer as timer

import numpy as np

import pyscf
import pyscf.cc
import pyscf.cc.dfccsd
import pyscf.pbc
import pyscf.pbc.cc

from vayesta.core.util import *
from vayesta.ewf import coupling
#from .solver import ClusterSolver
from .solver2 import ClusterSolver


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
        # EOM CCSD
        eom_ccsd: list = dataclasses.field(default_factory=list)
        eom_ccsd_nroots: int = 3
        # Tailored-CCSD
        tcc: bool = False
        tcc_fci_opts: dict = dataclasses.field(default_factory=dict)
        # Active space methods
        c_cas_occ: np.array = None
        c_cas_vir: np.array = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        solver_cls = self.get_solver_class()
        self.log.debug("CCSD PySCF class= %r" % solver_cls)
        frozen = self.cluster.get_frozen_indices()
        self.log.debugv("frozen= %r", frozen)
        mo_coeff = self.cluster.all.coeff
        solver = solver_cls(self.mf, mo_coeff=mo_coeff, mo_occ=self.mf.mo_occ, frozen=frozen)
        # Options
        if self.opts.maxiter is not None: solver.max_cycle = self.opts.maxiter
        if self.opts.conv_tol is not None: solver.conv_tol = self.opts.conv_tol
        if self.opts.conv_tol_normt is not None: solver.conv_tol_normt = self.opts.conv_tol_normt
        self.solver = solver

        # --- Results
        self.t1 = None
        self.t2 = None
        self.l1 = None
        self.l2 = None
        self.eris = None
        # TODO: REMOVE
        # EOM-CCSD
        self.ip_energy = None
        self.ip_coeff = None
        self.ea_energy = None
        self.ea_coeff = None
        # EE-EOM-CCSD
        self.ee_s_energy = None
        self.ee_t_energy = None
        self.ee_sf_energyy = None
        self.ee_s_coeff = None
        self.ee_t_coeff = None
        self.ee_sf_coeff = None

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
        self.t1 = None
        self.t2 = None
        self.l1 = None
        self.l2 = None
        self.eris = None
        # EOM-CCSD
        self.ip_energy = None
        self.ip_coeff = None
        self.ea_energy = None
        self.ea_coeff = None
        # EE-EOM-CCSD
        self.ee_s_energy = None
        self.ee_t_energy = None
        self.ee_sf_energyy = None
        self.ee_s_coeff = None
        self.ee_t_coeff = None
        self.ee_sf_coeff = None

    def get_t1(self):
        return self.t1

    def get_t2(self):
        return self.t2

    def get_c1(self, intermed_norm=True):
        """C1 in intermediate normalization."""
        if not intermed_norm:
            raise ValueError()
        return self.t1

    def get_c2(self, intermed_norm=True):
        """C2 in intermediate normalization."""
        if not intermed_norm:
            raise ValueError()
        return self.t2 + einsum('ia,jb->ijab', self.t1, self.t1)

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
        with log_time(self.log.timing, "Time for AO->MO transformation: %s"):
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
        if self.opts.v_ext is not None:
            self.log.debugv("Adding self.opts.v_ext to eris.fock")
            # Make sure there are no side effects:
            eris = copy.copy(eris)
            # Replace fock instead of modifying it!
            if self.is_rhf:
                eris.fock = (eris.fock + self.opts.v_ext)
            else:
                eris.focka = eris.fock[0] + self.opts.v_ext[0]
                eris.fockb = eris.fock[1] + self.opts.v_ext[1]
                eris.fock = (eris.focka, eris.fockb)

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
            self.log.error("%s not converged!", self.__class__.__name__)
        else:
            self.log.debugv("%s converged.", self.__class__.__name__)
        self.e_corr = self.solver.e_corr
        self.converged = self.solver.converged
        self.t1 = self.solver.t1
        self.t2 = self.solver.t2
        if self.is_rhf:
            self.log.debugv("tr(T1)= %.8f", np.trace(self.t1))
        else:
            self.log.debugv("tr(alpha-T1)= %.8f", np.trace(self.t1[0]))
            self.log.debugv("tr( beta-T1)= %.8f", np.trace(self.t1[1]))

        self.log.debug("Cluster: E(corr)= % 16.8f Ha", self.solver.e_corr)
        self.log.timing("Time for CCSD:  %s", time_string(timer()-t0))

        if hasattr(self.solver, '_norm_dt1'):
            self.log.debug("Tailored CC: |dT1|= %.2e |dT2|= %.2e", self.solver._norm_dt1, self.solver._norm_dt2)
            del self.solver._norm_dt1, self.solver._norm_dt2

        if t_diagnostic: self.t_diagnostic()

        if 'IP' in self.opts.eom_ccsd:
            self.ip_energy, self.ip_coeff = self.eom_ccsd('IP', eris)
        if 'EA' in self.opts.eom_ccsd:
            self.ea_energy, self.ea_coeff = self.eom_ccsd('EA', eris)
        if 'EE-S' in self.opts.eom_ccsd:
            self.ee_s_energy, self.ee_s_coeff = self.eom_ccsd('EE-S', eris)
        if 'EE-T' in self.opts.eom_ccsd:
            self.ee_t_energy, self.ee_t_coeff = self.eom_ccsd('EE-T', eris)
        if 'EE-SF' in self.opts.eom_ccsd:
            self.ee_sf_energy, self.ee_sf_coeff = self.eom_ccsd('EE-SF', eris)

    def solve_lambda(self, l1=None, l2=None, eris=None):
        if eris is None: eris = self.eris
        with log_time(self.log.info, "Time for lambda-equations: %s"):
            self.log.info("Solving lambda-equations %s initial guess...", "with" if (l2 is not None) else "without")
            self.l1, self.l2 = self.solver.solve_lambda(l1=l1, l2=l2, eris=eris)
            self.log.info("Lambda equations done. Lambda converged: %r", self.solver.converged_lambda)
            if not self.solver.converged_lambda:
                self.log.error("Solution of lambda-equation not converged!")
        return self.l1, self.l2

    def make_rdm1(self, l1=None, l2=None, with_frozen=False):
        if l1 is None: l1 = self.get_l1()
        if l2 is None: l2 = self.get_l2()
        self.log.debug("Making RDM1...")
        self.dm1 = self.solver.make_rdm1(l1=l1, l2=l2, with_frozen=with_frozen)
        return self.dm1

    def make_rdm2(self, l1=None, l2=None, with_frozen=False):
        if l1 is None: l1 = self.get_l1()
        if l2 is None: l2 = self.get_l2()
        self.log.debug("Making RDM2...")
        self.dm2 = self.solver.make_rdm2(l1=l1, l2=l2, with_frozen=with_frozen)
        return self.dm2

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


class UCCSD_Solver(CCSD_Solver):

    def get_solver_class(self):
        # No DF-UCCSD class in PySCF
        # Molecular UCCSD does not support DF either!
        if self.base.boundary_cond.startswith('periodic'):
            return pyscf.pbc.cc.ccsd.UCCSD
        return pyscf.cc.uccsd.UCCSD

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
        return None
