import dataclasses
from timeit import default_timer as timer

import numpy as np

import pyscf
import pyscf.cc
import pyscf.cc.dfccsd

from vayesta.core.util import *
from vayesta.ewf import coupling
from .solver import ClusterSolver


class CCSDSolver(ClusterSolver):

    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        # Convergence
        maxiter: int = 100              # Max number of iterations
        conv_etol: float = None     # Convergence energy tolerance
        conv_ttol: float = None     # Convergence amplitude tolerance
        #conv_etol: float = 1e-12        # Convergence energy tolerance
        #conv_ttol: float = 1e-10        # Convergence amplitude tolerance

        # Self-consistent mode
        sc_mode: int = NotSet
        # DM
        dm_with_frozen: bool = NotSet
        # EOM CCSD
        eom_ccsd: list = NotSet  # {'IP', 'EA', 'EE-S', 'EE-D', 'EE-SF'}
        eom_ccsd_nroots: int = NotSet
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


    def kernel(self, init_guess=None, eris=None, coupled_fragments=None, t_diagnostic=True):

        if coupled_fragments is None:
            coupled_fragments = self.fragment.opts.coupled_fragments

        # For 2D-systems the Coulomb repulsion is not PSD
        # Density-fitted CCSD does not support non-PSD three-center integrals,
        # thus we need a four-center formulation, where non PSD elements can be summed in
        if self.base.boundary_cond in ('periodic-1D', 'periodic-2D') or not hasattr(self.mf, 'with_df'):
            cls = pyscf.cc.ccsd.CCSD
        else:
            cls = pyscf.cc.dfccsd.RCCSD
        self.log.debug("CCSD class= %r" % cls)
        cc = cls(self.mf, mo_coeff=self.mo_coeff, mo_occ=self.mo_occ, frozen=self.get_frozen_indices())
        if self.opts.maxiter is not None: cc.max_cycle = self.opts.maxiter
        if self.opts.conv_etol is not None: cc.conv_tol = self.opts.conv_etol
        if self.opts.conv_ttol is not None: cc.conv_tol_normt = self.opts.conv_ttol

        # Integral transformation
        if eris is None:
            t0 = timer()
            eris = self.base.get_eris(cc)
            self.log.timing("Time for AO->MO of (ij|kl):  %s", time_string(timer()-t0))
        #else:
        #    # DEBUG:
        #    eris = self.base.get_eris(cc)
        #    for kind in ["oooo", "ovoo", "ovvo", "oovv", "ovov", "ovvv", "vvvv"]:
        #        diff = getattr(self._eris, kind) - getattr(eris, kind)
        #        log.debug("Difference (%2s|%2s): max= %.2e norm= %.2e", kind[:2], kind[2:], abs(diff).max(), np.linalg.norm(diff))
        self.log.debugv("eris.mo_energy:\n%r", eris.mo_energy)
        self.log.debugv("sum(eris.mo_energy)= %.8e", sum(eris.mo_energy))
        self.log.debugv("Tr(eris.Fock)= %.8e", np.trace(eris.fock))

        # Tailored CC
        if self.opts.tcc:
            self.log.info("Adding tailor function to CCSD.")
            cc.tailor_func = coupling.make_cas_tcc_function(
                    self, c_cas_occ=self.opts.c_cas_occ, c_cas_vir=self.opts.c_cas_vir, eris=eris).__get__(cc)

        elif self.opts.sc_mode and self.base.iteration > 1:
            # __get__(cc) to bind the tailor function as a method,
            # rather than just a callable attribute
            self.log.info("Adding tailor function to CCSD.")
            cc.tailor_func = coupling.make_cross_fragment_tcc_function(self, mode=self.opts.sc_mode).__get__(cc)

        # This should include the SC mode?
        elif coupled_fragments and np.all([x.results is not None for x in coupled_fragments]):
            self.log.info("Adding tailor function to CCSD.")
            cc.tailor_func = coupling.make_cross_fragment_tcc_function(self, mode=(self.opts.sc_mode or 3),
                    coupled_fragments=coupled_fragments).__get__(cc)

        t0 = timer()
        if init_guess:
            self.log.info("Running CCSD with initial guess for %r..." % list(init_guess.keys()))
            cc.kernel(eris=eris, **init_guess)
        else:
            self.log.info("Running CCSD...")
            cc.kernel(eris=eris)
        (self.log.info if cc.converged else self.log.error)("CCSD done. converged: %r", cc.converged)
        self.log.debug("E(full corr)= % 16.8f Ha", cc.e_corr)
        self.log.timing("Time for CCSD:  %s", time_string(timer()-t0))

        if hasattr(cc, '_norm_dt1'):
            self.log.debug("Tailored CC: |dT1|= %.2e |dT2|= %.2e", cc._norm_dt1, cc._norm_dt2)
            del cc._norm_dt1
            del cc._norm_dt2

        if t_diagnostic:
            self.t_diagnostic(cc)

        results = self.Results(
                converged=cc.converged, e_corr=cc.e_corr, c_occ=self.c_active_occ, c_vir=self.c_active_vir, eris=eris,
                t1=cc.t1, t2=cc.t2)

        solve_lambda = (self.opts.make_rdm1 or self.opts.make_rdm2)
        if solve_lambda:
            t0 = timer()
            self.log.info("Solving lambda equations...")
            # This automatically sets cc.l1, cc.l2:
            results.l1, results.l2 = cc.solve_lambda(cc.t1, cc.t2, eris=eris)
            self.log.info("Lambda equations done. Lambda converged: %r", cc.converged_lambda)
            if not cc.converged_lambda:
                self.log.error("Solution of lambda equation not converged!")
            self.log.timing("Time for lambda-equations: %s", time_string(timer()-t0))

        if self.opts.make_rdm1:
            self.log.info("Making RDM1...")
            #results.dm1 = cc.make_rdm1(eris=eris, ao_repr=True)
            results.dm1 = cc.make_rdm1(with_frozen=self.opts.dm_with_frozen)
        if self.opts.make_rdm2:
            self.log.info("Making RDM2...")
            results.dm2 = cc.make_rdm2(with_frozen=self.opts.dm_with_frozen)

        def run_eom_ccsd(kind, nroots=None):
            nroots = nroots or self.opts.eom_ccsd_nroots
            kind = kind.upper()
            assert kind in ('IP', 'EA', 'EE-S', 'EE-T', 'EE-SF')
            self.log.info("Running %s-EOM-CCSD (nroots=%d)...", kind, nroots)
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

        if 'IP' in self.opts.eom_ccsd:
            results.ip_energy, results.ip_coeff = run_eom_ccsd('IP')
        if 'EA' in self.opts.eom_ccsd:
            results.ea_energy, results.ea_coeff = run_eom_ccsd('EA')
        if 'EE-S' in self.opts.eom_ccsd:
            results.ee_s_energy, results.ee_s_coeff = run_eom_ccsd('EE-S')
        if 'EE-T' in self.opts.eom_ccsd:
            results.ee_t_energy, results.ee_t_coeff = run_eom_ccsd('EE-T')
        if 'EE-SF' in self.opts.eom_ccsd:
            results.ee_sf_energy, results.ee_sf_coeff = run_eom_ccsd('EE-SF')


        return results



    def t_diagnostic(self, cc):
        self.log.info("T-Diagnostic")
        self.log.info("------------")
        try:
            dg_t1 = cc.get_t1_diagnostic()
            dg_d1 = cc.get_d1_diagnostic()
            dg_d2 = cc.get_d2_diagnostic()
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
