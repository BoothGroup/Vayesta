import dataclasses
import numpy as np
from .solver import ClusterSolver, UClusterSolver
from .cisd import CISD_Solver
from vayesta.core.types import CCSD_WaveFunction
from vayesta.core.util import *

from . import coupling

import pyscf.cc
from ._uccsd_eris import uao2mo


class RCCSD_Solver(ClusterSolver):
    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        # Convergence
        max_cycle: int = 100  # Max number of iterations
        conv_tol: float = None  # Convergence energy tolerance
        conv_tol_normt: float = None  # Convergence amplitude tolerance
        init_guess: str = 'MP2'         # ['MP2', 'CISD']
        solve_lambda: bool = False  # If false, use Lambda=T approximation
        # Self-consistent mode
        sc_mode: int = None

    def kernel(self, t1=None, t2=None, l1=None, l2=None, coupled_fragments=None, t_diagnostic=True):
        mf_clus, frozen = self.hamil.to_pyscf_mf(allow_dummy_orbs=True, allow_df=True)
        solver_cls = self.get_solver_class(mf_clus)
        self.log.debugv("PySCF solver class= %r" % solver_cls)
        mycc = solver_cls(mf_clus, frozen=frozen)

        if self.opts.max_cycle is not None:
            mycc.max_cycle = self.opts.max_cycle
        if self.opts.conv_tol is not None:
            mycc.conv_tol = self.opts.conv_tol
        if self.opts.conv_tol_normt is not None:
            mycc.conv_tol_normt = self.opts.conv_tol_normt
        mycc.callback = self.get_callback()

        #if self.opts.sc_mode and self.base.iteration > 1:
        #    self.set_callback(coupling.make_cross_fragment_tcc_function(self, mode=self.opts.sc_mode))
        # This should include the SC mode?
        #elif coupled_fragments and np.all([x.results is not None for x in coupled_fragments]):
        #    self.set_callback(coupling.make_cross_fragment_tcc_function(self, mode=(self.opts.sc_mode or 3),
        #                                                                coupled_fragments=coupled_fragments))

        if t1 is None and t2 is None:
            t1, t2 = self.generate_init_guess()

        self.log.info("Solving CCSD-equations %s initial guess...", "with" if (t2 is not None) else "without")

        mycc.kernel(t1=t1, t2=t2)
        self.converged = mycc.converged

        if t_diagnostic:
            self.t_diagnostic(mycc)

        self.print_extra_info(mycc)


        if self.opts.solve_lambda:
            mycc.solve_lambda(l1=l1, l2=l2)
            self.converged = self.converged and mycc.converged_lambda

        self.wf = CCSD_WaveFunction(self.hamil.mo, mycc.t1, mycc.t2, l1=mycc.l1, l2=mycc.l2)

    def get_solver_class(self, mf):
        if hasattr(mf, "with_df") and mf.with_df is not None:
            return pyscf.cc.dfccsd.RCCSD
        return pyscf.cc.ccsd.RCCSD

    def generate_init_guess(self, eris=None):
        if self.opts.init_guess in ('default', 'MP2'):
            # CCSD will build MP2 amplitudes
            return None, None
        if self.opts.init_guess == 'CISD':
            cisd = CISD_Solver(self.hamil)
            cisd.kernel()
            wf = cisd.wf.as_ccsd()
            return wf.t1, wf.t2
        raise ValueError("init_guess= %r" % self.opts.init_guess)

    @log_method()
    def t_diagnostic(self, solver):
        self.log.info("T-Diagnostic")
        self.log.info("------------")
        try:
            dg_t1 = solver.get_t1_diagnostic()
            dg_d1 = solver.get_d1_diagnostic()
            dg_d2 = solver.get_d2_diagnostic()
            dg_t1_msg = "good" if dg_t1 <= 0.02 else "inadequate!"
            dg_d1_msg = "good" if dg_d1 <= 0.02 else ("fair" if dg_d1 <= 0.05 else "inadequate!")
            dg_d2_msg = "good" if dg_d2 <= 0.15 else ("fair" if dg_d2 <= 0.18 else "inadequate!")
            fmt = 3 * "  %2s= %.3f (%s)"
            self.log.info(fmt, "T1", dg_t1, dg_t1_msg, "D1", dg_d1, dg_d1_msg, "D2", dg_d2, dg_d2_msg)
            # good: MP2~CCSD~CCSD(T) / fair: use MP2/CCSD with caution
            self.log.info("  (T1<0.02: good / D1<0.02: good, D1<0.05: fair / D2<0.15: good, D2<0.18: fair)")
            if dg_t1 > 0.02 or dg_d1 > 0.05 or dg_d2 > 0.18:
                self.log.info("  at least one diagnostic indicates that CCSD is not adequate!")
        except Exception as e:
            self.log.error("Exception in T-diagnostic: %s", e)

    def get_callback(self):
        return None

    def print_extra_info(self, mycc):
        pass

    def couple_iterations(self, fragments):
        self.set_callback(coupling.couple_ccsd_iterations(self, fragments))

    def _debug_exact_wf(self, wf):
        mo = self.hamil.mo
        # Project onto cluster:
        ovlp = self.hamil._fragment.base.get_ovlp()
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
        mo = self.hamil.mo
        t1 = np.random.rand(mo.nocc, mo.nvir)
        l1 = np.random.rand(mo.nocc, mo.nvir)
        t2 = np.random.rand(mo.nocc, mo.nocc, mo.nvir, mo.nvir)
        l2 = np.random.rand(mo.nocc, mo.nocc, mo.nvir, mo.nvir)
        self.wf = CCSD_WaveFunction(mo, t1, t2, l1=l1, l2=l2)
        self.converged = True


class UCCSD_Solver(UClusterSolver, RCCSD_Solver):
    @dataclasses.dataclass
    class Options(RCCSD_Solver.Options):
        pass

    def get_solver_class(self, mf):
        return UCCSD

    def t_diagnostic(self, solver):
        """T diagnostic not implemented for UCCSD in PySCF."""
        self.log.info("T diagnostic not implemented for UCCSD in PySCF.")

    def _debug_exact_wf(self, wf):
        mo = self.hamil.mo
        # Project onto cluster:
        ovlp = self.hamil._fragment.base.get_ovlp()
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


# Subclass pyscf UCCSD to enable support of spin-dependent ERIs.
class UCCSD(pyscf.cc.uccsd.UCCSD):
    ao2mo = uao2mo
