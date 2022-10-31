import dataclasses
import copy
import typing
from typing import Optional, List
from timeit import default_timer as timer

import numpy as np

import pyscf
import pyscf.cc
import pyscf.cc.dfccsd
import pyscf.pbc
import pyscf.pbc.cc

from vayesta.core.types import Orbitals
from vayesta.core.types import WaveFunction
from vayesta.core.types import CCSD_WaveFunction
from vayesta.core.qemb import scrcoulomb
from vayesta.core.ao2mo import helper
from vayesta.core.util import *
from .solver import ClusterSolver

from . import coupling
from . import tccsd


class CCSD_Solver(ClusterSolver):

    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        # Convergence
        max_cycle: int = 100            # Max number of iterations
        conv_tol: float = None          # Convergence energy tolerance
        conv_tol_normt: float = None    # Convergence amplitude tolerance
        init_guess: str = 'MP2'         # ['MP2', 'CISD']
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
        # Tailor/externally correct CCSD with other fragments
        external_corrections: Optional[List[typing.Any]] = dataclasses.field(default_factory=list)
        # Lambda equations
        solve_lambda: bool = True

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
        if self.opts.max_cycle is not None: solver.max_cycle = self.opts.max_cycle
        if self.opts.conv_tol is not None: solver.conv_tol = self.opts.conv_tol
        if self.opts.conv_tol_normt is not None: solver.conv_tol_normt = self.opts.conv_tol_normt
        self.solver = solver
        self.eris = None

    def get_solver_class(self):
        # For 2D-systems the Coulomb repulsion is not PSD
        # Density-fitted CCSD does not support non-PSD three-center integrals,
        # thus we need a four-center formulation, where non PSD elements can be summed in
        if self.base.pbc_dimension in (1,2):
            return pyscf.cc.ccsd.CCSD
        if hasattr(self.mf, 'with_df') and self.mf.with_df is not None:
            return pyscf.cc.dfccsd.RCCSD
        return pyscf.cc.ccsd.CCSD

    def get_cisd_solver(self):
        from vayesta.solver import CISD_Solver
        return CISD_Solver

    def reset(self):
        super().reset()
        self.eris = None

    def get_eris(self):
        self.log.debugv("Getting ERIs for type(self.solver)= %r", type(self.solver))
        with log_time(self.log.timing, "Time for 2e-integral transformation: %s"):
            self.eris = self.base.get_eris_object(self.solver)
        return self.eris

    def get_init_guess(self, eris=None):
        if self.opts.init_guess in ('default', 'MP2'):
            # CCSD will build MP2 amplitudes
            return None, None
        if self.opts.init_guess == 'CISD':
            cisd = self.get_cisd_solver()(self.mf, self.fragment, self.cluster)
            cisd.kernel(eris=eris)
            wf = cisd.wf.as_ccsd()
            return wf.t1, wf.t2
        raise ValueError("init_guess= %r" % self.opts.init_guess)

    def add_screening(self, *args, **kwargs):
        raise NotImplementedError("Screening only implemented for unrestricted spin-symmetry.")

    def add_potential(self, eris, v_ext):
        self.log.debugv("Adding self.v_ext to eris.fock")
        # Make sure there are no side effects:
        eris = copy.copy(eris)
        # Replace fock instead of modifying it!
        if self.is_rhf:
            eris.fock = (eris.fock + v_ext)
        else:
            eris.focka = eris.fock[0] + v_ext[0]
            eris.fockb = eris.fock[1] + v_ext[1]
            eris.fock = (eris.focka, eris.fockb)
        return eris

    def kernel(self, t1=None, t2=None, eris=None, l1=None, l2=None, seris_ov=None, coupled_fragments=None, t_diagnostic=True):
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
        if eris is None:
            with log_time(self.log.info, "Time for ERIs: %s"):
                eris = self.get_eris()

        # Add screening [optional]
        if seris_ov is not None:
            eris = self.add_screening(eris, seris_ov)

        # Add additional potential [optional]
        if self.v_ext is not None:
            eris = self.add_potential(eris, self.v_ext)

        # Tailored CC "solver"
        if self.opts.tcc:
            if self.spinsym == 'unrestricted':
                raise NotImplementedError("TCCSD for unrestricted spin-symmetry")
            self.set_callback(tccsd.make_cas_tcc_function(
                              self, c_cas_occ=self.opts.c_cas_occ, c_cas_vir=self.opts.c_cas_vir, eris=eris))
        # Tailoring or external corrections from other fragments
        elif self.opts.external_corrections:
            # Tailoring of T1 and T2
            tailors = [ec for ec in self.opts.external_corrections if (ec[1] == 'tailor')]
            externals = [ec for ec in self.opts.external_corrections if (ec[1] == 'external')]
            if tailors and externals:
                raise NotImplementedError
            if tailors:
                tailor_frags = self.base.get_fragments(id=[t[0] for t in tailors])
                proj = tailors[0][2]
                if np.any([(t[2] != proj) for t in tailors]):
                    raise NotImplementedError
                self.log.info("Tailoring CCSD from %d fragments (projectors= %d)", len(tailor_frags), proj)
                self.set_callback(coupling.tailor_with_fragments(self, tailor_frags, project=proj))
            # External correction of T1 and T2
            if externals:
                self.log.info("Externally correct CCSD from %d fragments", len(externals))
                self.set_callback(coupling.externally_correct(self, externals))

        elif self.opts.sc_mode and self.base.iteration > 1:
            raise NotImplementedError
            self.set_callback(coupling.make_cross_fragment_tcc_function(self, mode=self.opts.sc_mode))
        # This should include the SC mode?
        elif coupled_fragments and np.all([x.results is not None for x in coupled_fragments]):
            raise NotImplementedError
            self.set_callback(coupling.make_cross_fragment_tcc_function(self, mode=(self.opts.sc_mode or 3),
                              coupled_fragments=coupled_fragments))

        # Initial guess
        if t1 is not None or t2 is not None:
            self.log.info("Solving CCSD-equations with initial guess...")
        else:
            t1, t2 = self.get_init_guess(eris=eris)

        with log_time(self.log.info, "Time for T-equations: %s"):
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

        if hasattr(self.solver, '_norm_dt1'):
            self.log.debug("Tailored CC: |dT1|= %.2e |dT2|= %.2e", self.solver._norm_dt1, self.solver._norm_dt2)
            del self.solver._norm_dt1, self.solver._norm_dt2

        if self.opts.solve_lambda:
            self.log.info("Solving lambda-equations with%s initial guess...", ("out" if (l2 is None) else ""))
            with log_time(self.log.info, "Time for Lambda-equations: %s"):
                l1, l2 = self.solver.solve_lambda(l1=l1, l2=l2, eris=eris)
            if not self.solver.converged_lambda:
                self.log.error("Lambda-equations not converged!")
        else:
            self.log.info("Using Lambda=T approximation for Lambda-amplitudes.")
            l1, l2 = self.solver.t1, self.solver.t2

        # Remove screening (for energy calculation etc)
        if hasattr(eris, 'restore_bare'):
            eris.restore_bare()

        if t_diagnostic: self.t_diagnostic()

        mo = Orbitals(self.cluster.c_active, occ=self.cluster.nocc_active)
        self.wf = CCSD_WaveFunction(mo, self.solver.t1, self.solver.t2, l1=l1, l2=l2)

    @log_method()
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

    def set_callback(self, callback):
        if not hasattr(self.solver, 'callback'):
            raise AttributeError("CCSD does not have attribute 'callback'.")
        self.log.debug("Adding callback function to CCSD.")
        self.solver.callback = callback

    def couple_iterations(self, fragments):
        self.set_callback(coupling.couple_ccsd_iterations(self, fragments))

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
        self.wf = CCSD_WaveFunction(mo, t1, t2, l1=l1, l2=l2)
        self.converged = True

    def _debug_random_wf(self):
        mo = Orbitals(self.cluster.c_active, occ=self.cluster.nocc_active)
        t1 = np.random.rand(mo.nocc, mo.nvir)
        l1 = np.random.rand(mo.nocc, mo.nvir)
        t2 = np.random.rand(mo.nocc, mo.nocc, mo.nvir, mo.nvir)
        l2 = np.random.rand(mo.nocc, mo.nocc, mo.nvir, mo.nvir)
        self.wf = CCSD_WaveFunction(mo, t1, t2, l1=l1, l2=l2)
        self.converged = True


class UCCSD_Solver(CCSD_Solver):

    def get_solver_class(self):
        # No DF-UCCSD class in PySCF
        # Molecular UCCSD does not support DF either!
        if self.base.pbc_dimension > 0:
            return pyscf.pbc.cc.ccsd.UCCSD
        return pyscf.cc.uccsd.UCCSD

    def get_cisd_solver(self):
        from vayesta.solver import UCISD_Solver
        return UCISD_Solver

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

    def add_screening(self, eris, seris_ov):
        seris = scrcoulomb.get_screened_eris_ccsd(eris, seris_ov)
        # Correct virtual-virtual cluster Fock matrix:
        nocca, noccb = eris.nocc
        gaa, gab, gbb = seris.get_bare()
        saa, sab, sbb = seris_ov
        dfvva = -einsum('ipiq->pq', (saa-gaa))
        dfvvb = -einsum('ipiq->pq', (sbb-gbb))
        va = np.s_[nocca:]
        vb = np.s_[noccb:]
        seris.focka[va,va] += dfvva
        seris.fockb[vb,vb] += dfvvb
        seris.fock = (seris.focka, seris.fockb)
        return seris

    def t_diagnostic(self):
        """T diagnostic not implemented for UCCSD in PySCF."""
        self.log.info("T diagnostic not implemented for UCCSD in PySCF.")

    def _debug_exact_wf(self, wf):
        mo = Orbitals(self.cluster.c_active, occ=self.cluster.nocc_active)
        # Project onto cluster:
        ovlp = self.fragment.base.get_ovlp()
        roa = dot(wf.mo.coeff_occ[0].T, ovlp, mo.coeff_occ[0])
        rob = dot(wf.mo.coeff_occ[1].T, ovlp, mo.coeff_occ[1])
        rva = dot(wf.mo.coeff_vir[0].T, ovlp, mo.coeff_vir[0])
        rvb = dot(wf.mo.coeff_vir[1].T, ovlp, mo.coeff_vir[1])
        t1a = dot(roa.T, wf.t1a, rva)
        t1b = dot(rob.T, wf.t1b, rvb)
        t2aa = einsum('Ii,Jj,IJAB,Aa,Bb->ijab', roa, roa, wf.t2aa, rva, rva)
        t2ab = einsum('Ii,Jj,IJAB,Aa,Bb->ijab', roa, rob, wf.t2ab, rva, rvb)
        t2bb = einsum('Ii,Jj,IJAB,Aa,Bb->ijab', rob, rob, wf.t2bb, rvb, rvb)
        t1 = (t1a, t1b)
        t2 = (t2aa, t2ab, t2bb)
        if wf.l1 is not None:
            l1a = dot(roa.T, wf.l1a, rva)
            l1b = dot(rob.T, wf.l1b, rvb)
            l2aa = einsum('Ii,Jj,IJAB,Aa,Bb->ijab', roa, roa, wf.l2aa, rva, rva)
            l2ab = einsum('Ii,Jj,IJAB,Aa,Bb->ijab', roa, rob, wf.l2ab, rva, rvb)
            l2bb = einsum('Ii,Jj,IJAB,Aa,Bb->ijab', rob, rob, wf.l2bb, rvb, rvb)
            l1 = (l1a, l1b)
            l2 = (l2aa, l2ab, l2bb)
        else:
            l1 = l2 = None
        self.wf = CCSD_WaveFunction(mo, t1, t2, l1=l1, l2=l2)
        self.converged = True

    def _debug_random_wf(self):
        raise NotImplementedError
