import dataclasses
from timeit import default_timer as timer

import numpy as np

import pyscf
import pyscf.ao2mo
import pyscf.cc
import pyscf.cc.dfccsd
import pyscf.pbc
import pyscf.ci
import pyscf.mcscf
import pyscf.fci
import pyscf.fci.addons
import pyscf.fci.direct_spin0
import pyscf.fci.direct_spin1

from vayesta.core.util import *


def get_solver_class(solver):
    #raise NotImplementedError
    if solver.upper() in ('CCSD', 'CCSD(T)', 'TCCSD'):
        return CCSDSolver
    if solver.upper() == 'FCI':
        return FCISolver
    raise NotImplementedError("Unknown solver %s" % solver)

@dataclasses.dataclass
class ClusterSolverOptions(Options):
    eom_ccsd: bool = NotSet
    make_rdm1: bool = True
    # CCSD specific
    # Active space methods
    c_cas_occ: np.array = None
    c_cas_vir: np.array = None
    # Tailored-CCSD
    tcc: bool = False
    tcc_spin: float = None


@dataclasses.dataclass
class ClusterSolverResults:
    converged: bool = False
    e_corr: float = 0.0
    c_occ: np.array = None
    c_vir: np.array = None
    # Density matrix in MO representation:
    dm1: np.array = None
    eris: 'typing.Any' = None


class ClusterSolver:
    """Base class for cluster solver"""

    def __init__(self, fragment, mo_coeff, mo_occ, nocc_frozen, nvir_frozen,
            options=None, log=None, **kwargs):
        """

        Arguments
        ---------
        nocc_frozen : int
            Number of frozen occupied orbitals. Need to be at the start of mo_coeff.
        nvir_frozen : int
            Number of frozen virtual orbitals. Need to be at the end of mo_coeff.
        """
        self.log = log or fragment.log
        self.fragment = fragment
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.nocc_frozen = nocc_frozen
        self.nvir_frozen = nvir_frozen

        if options is None:
            options = ClusterSolverOptions(**kwargs)
        else:
            options = options.replace(kwargs)
        options = options.replace(self.base.opts, select=NotSet)
        self.opts = options


    @property
    def base(self):
        return self.fragment.base

    @property
    def mf(self):
        return self.fragment.mf

    @property
    def mol(self):
        return self.fragment.mol

    @property
    def nmo(self):
        """Total number of MOs (not just active)."""
        return self.mo_coeff.shape[-1]

    @property
    def nocc(self):
        return np.count_nonzero(self.mo_occ > 0)

    @property
    def nactive(self):
        """Number of active MOs."""
        return self.nmo - self.nfrozen

    @property
    def nfrozen(self):
        return self.nocc_frozen + self.nvir_frozen

    def get_active_slice(self):
        #slc = np.s_[self.nocc_frozen:-self.nvir_frozen]
        slc = np.s_[self.nocc_frozen:self.nocc_frozen+self.nactive]
        return slc

    def get_frozen_indices(self):
        nmo = self.mo_coeff.shape[-1]
        idx = list(range(self.nocc_frozen)) + list(range(nmo-self.nvir_frozen, nmo))
        return idx

    @property
    def c_active_occ(self):
        """Active occupied orbital coefficients."""
        return self.mo_coeff[:,self.nocc_frozen:self.nocc]

    @property
    def c_active_vir(self):
        """Active virtual orbital coefficients."""
        #return self.mo_coeff[:,self.nocc:-self.nvir_frozen]
        return self.mo_coeff[:,self.nocc:self.nocc_frozen+self.nactive]


@dataclasses.dataclass
class CCSDSolverResults(ClusterSolverResults):
    t1: np.array = None
    t2: np.array = None
    l1: np.array = None
    l2: np.array = None
    # EOM-CCSD
    ip_energy: np.array = None
    ip_coeff: np.array = None
    ea_energy: np.array = None
    ea_coeff: np.array = None


class CCSDSolver(ClusterSolver):

    def kernel(self, init_guess=None, eris=None, t_diagnostic=True):

        # Do not use pbc.ccsd for Gamma point CCSD -> always use molecular code
        #if self.base.boundary_cond == 'open':
        #    cls = pyscf.cc.CCSD
        #else:
        #    import pyscf.pbc.cc
        #    cls = pyscf.pbc.cc.CCSD
        #    #cls = pyscf.cc.ccsd.CCSD

        # For 2D-systems the Coulomb repulsion is not PSD
        # Density-fitted CCSD does not support non-PSD three-center integrals,
        # thus we need a four-center formulation, where non PSD elements can be summed in
        if self.base.boundary_cond in ('periodic-1D', 'periodic-2D') or not hasattr(self.mf, 'with_df'):
            cls = pyscf.cc.ccsd.CCSD
        else:
            cls = pyscf.cc.dfccsd.RCCSD

        self.log.debug("CCSD class= %r" % cls)
        cc = cls(self.mf, mo_coeff=self.mo_coeff, mo_occ=self.mo_occ, frozen=self.get_frozen_indices())

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

        results = CCSDSolverResults(
                converged=cc.converged, e_corr=cc.e_corr, c_occ=self.c_active_occ, c_vir=self.c_active_vir, eris=eris,
                t1=cc.t1, t2=cc.t2)

        solve_lambda = self.opts.make_rdm1
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
            results.dm1 = cc.make_rdm1(with_frozen=False)

        def eom_ccsd(kind, nroots=3):
            kind = kind.upper()
            assert kind in ("IP", "EA")
            self.log.info("Running %s-EOM-CCSD (nroots=%d)...", kind, nroots)
            eom_funcs = {"IP" : cc.ipccsd , "EA" : cc.eaccsd}
            t0 = timer()
            e, c = eom_funcs[kind](nroots=nroots, eris=eris)
            self.log.timing("Time for %s-EOM-CCSD:  %s", kind, time_string(timer()-t0))
            if nroots == 1:
                e, c = [e], [c]
            return e, c

        if self.opts.eom_ccsd in (True, "IP"):
            results.ip_energy, results.ip_coeff = eom_ccsd("IP")
        if self.opts.eom_ccsd in (True, "EA"):
            results.ea_energy, results.ea_coeff = eom_ccsd("EA")

        return results

    def t_diagnostic(self, cc):
        self.log.info("T-Diagnostic")
        self.log.info("************")
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



@dataclasses.dataclass
class CISolverResults(ClusterSolverResults):
    # CI coefficients
    c0: float = None
    c1: np.array = None
    c2: np.array = None


class FCISolver(ClusterSolver):

    def kernel(self, init_guess=None, eris=None):
        """TODO: Avoid CASCI and use FCISolver, to use unfolded PBC eris."""
        nelec = sum(self.mo_occ[self.get_active_slice()])
        casci = pyscf.mcscf.CASCI(self.mf, self.nactive, nelec)
        casci.canonicalization = False

        self.log.debug("Running CASCI with (%d, %d) CAS", nelec, self.nactive)
        t0 = timer()
        e_tot, e_cas, wf, *_ = casci.kernel(mo_coeff=self.mo_coeff)
        self.log.debug("FCI done. converged: %r", casci.converged)
        self.log.timing("Time for FCI: %s", time_string(timer()-t0))
        e_corr = (e_tot-self.mf.e_tot)

        cisdvec = pyscf.ci.cisd.from_fcivec(wf, self.nactive, nelec)
        nocc = nelec // 2
        c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, self.nactive, nocc)

        # Temporary workaround (eris needed for energy later)
        class ERIs:
            pass
        eris = ERIs()
        c_act = self.mo_coeff[:,self.get_active_slice()]
        eris.fock = np.linalg.multi_dot((c_act.T, self.base.get_fock(), c_act))
        g = pyscf.ao2mo.full(self.mf._eri, c_act)
        o = np.s_[:nocc]
        v = np.s_[nocc:]
        eris.ovvo = pyscf.ao2mo.restore(1, g, self.nactive)[o,v,v,o]

        results = CISolverResults(
                converged=casci.converged, e_corr=e_corr, c_occ=self.c_active_occ, c_vir=self.c_active_vir, eris=eris,
                c0=c0, c1=c1, c2=c2)

        return results
