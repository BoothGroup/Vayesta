import dataclasses
from timeit import default_timer as timer

import numpy as np

import pyscf
import pyscf.ao2mo
import pyscf.cc
import pyscf.cc.dfccsd
import pyscf.pbc
import pyscf.ci
import pyscf.fci
import pyscf.fci.addons
import pyscf.fci.direct_spin0
import pyscf.fci.direct_spin1

from vayesta.core.util import *
from . import helper


def get_solver_class(solver):
    if solver.upper() in ('CCSD', 'CCSD(T)', 'TCCSD'):
        return CCSDSolver
    if solver.upper() == 'FCI':
        return FCISolver
    raise NotImplementedError("Unknown solver %s" % solver)


@dataclasses.dataclass
class ClusterSolverOptions(Options):
    eom_ccsd: bool = NotSet
    make_rdm1: bool = NotSet
    sc_mode: int = NotSet
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

        # Tailored CC
        if self.opts.tcc:
            self.log.info("Adding tailor function to CCSD.")
            cc.tailor_func = self.make_tcc_tailor_function(
                    c_cas_occ=self.opts.c_cas_occ, c_cas_vir=self.opts.c_cas_vir, eris=eris).__get__(cc)

        elif self.opts.sc_mode and self.base.iteration > 1:
            # __get__(cc) to bind the tailor function as a method,
            # rather than just a callable attribute
            self.log.info("Adding tailor function to CCSD.")
            cc.tailor_func = self.make_sc_tailor_function(mode=self.opts.sc_mode).__get__(cc)

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

        results = CCSDSolverResults(converged=cc.converged, e_corr=cc.e_corr, t1=cc.t1, t2=cc.t2,
                c_occ=self.c_active_occ, c_vir=self.c_active_vir, eris=eris)

        if self.opts.make_rdm1:
            try:
                t0 = timer()
                self.log.info("Making RDM1...")
                #results.dm1 = cc.make_rdm1(eris=eris, ao_repr=True)
                results.dm1 = cc.make_rdm1(eris=eris, with_frozen=False)
                self.log.info("RDM1 done. Lambda converged: %r", cc.converged_lambda)
                if not cc.converged_lambda:
                    self.log.error("Solution of lambda equation not converged!")
                self.log.timing("Time for RDM1: %s", time_string(timer()-t0))
            except Exception as e:
                self.log.error("Exception while making RDM1: %s", e)

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


    def make_tcc_tailor_function(self, c_cas_occ, c_cas_vir, eris):
        """Make tailor function for Tailored CC."""

        ncasocc = c_cas_occ.shape[-1]
        ncasvir = c_cas_vir.shape[-1]
        ncas = ncasocc + ncasvir
        nelec = 2*ncasocc

        self.log.info("Running FCI in (%d, %d) CAS", nelec, ncas)

        c_cas = np.hstack((c_cas_occ, c_cas_vir))
        ovlp = self.base.get_ovlp()

        # Rotation & projection into CAS
        ro = np.linalg.multi_dot((self.c_active_occ.T, ovlp, c_cas_occ))
        rv = np.linalg.multi_dot((self.c_active_vir.T, ovlp, c_cas_vir))
        r = np.block([[ro, np.zeros((ro.shape[0], rv.shape[1]))],
                    [np.zeros((rv.shape[0], ro.shape[1])), rv]])

        o = np.s_[:ncasocc]
        v = np.s_[ncasocc:]

        def make_cas_eris():
            """Make 4c ERIs in CAS."""
            t0 = timer()
            g_cas = np.zeros(4*[ncas])
            # 0 v
            g_cas[o,o,o,o] = einsum('IJKL,Ii,Jj,Kk,Ll->ijkl', eris.oooo[:], ro, ro, ro, ro)
            # 1 v
            g_cas[o,v,o,o] = einsum('IJKL,Ii,Jj,Kk,Ll->ijkl', eris.ovoo[:], ro, rv, ro, ro)
            g_cas[v,o,o,o] = g_cas[o,v,o,o].transpose(1,0,3,2)
            g_cas[o,o,o,v] = g_cas[o,v,o,o].transpose(2,3,0,1)
            g_cas[o,o,v,o] = g_cas[o,o,o,v].transpose(1,0,3,2)
            # 2 v
            g_cas[o,o,v,v] = einsum('IJKL,Ii,Jj,Kk,Ll->ijkl', eris.oovv[:], ro, ro, rv, rv)
            g_cas[v,v,o,o] = g_cas[o,o,v,v].transpose(2,3,0,1)
            g_cas[o,v,o,v] = einsum('IJKL,Ii,Jj,Kk,Ll->ijkl', eris.ovov[:], ro, rv, ro, rv)
            g_cas[v,o,v,o] = g_cas[o,v,o,v].transpose(1,0,3,2)
            g_cas[o,v,v,o] = einsum('IJKL,Ii,Jj,Kk,Ll->ijkl', eris.ovvo[:], ro, rv, rv, ro)
            g_cas[v,o,o,v] = g_cas[o,v,v,o].transpose(2,3,0,1)
            # 3 v
            nocc = self.c_active_occ.shape[-1]
            nvir = self.c_active_vir.shape[-1]
            if eris.ovvv.ndim == 3:
                nvir_pair = nvir*(nvir+1)//2
                g_ovvv = pyscf.lib.unpack_tril(eris.ovvv.reshape(nocc*nvir, nvir_pair)).reshape(nocc,nvir,nvir,nvir)
            else:
                g_ovvv = eris.ovvv[:]
            g_cas[o,v,v,v] = einsum('IJKL,Ii,Jj,Kk,Ll->ijkl', g_ovvv, ro, rv, rv, rv)
            g_cas[v,o,v,v] = g_cas[o,v,v,v].transpose(1,0,3,2)
            g_cas[v,v,o,v] = g_cas[o,v,v,v].transpose(2,3,0,1)
            g_cas[v,v,v,o] = g_cas[v,v,o,v].transpose(1,0,3,2)
            # 4 v
            if hasattr(eris, 'vvvv') and eris.vvvv is not None:
                if eris.vvvv.ndim == 2:
                    g_vvvv = pyscf.ao2mo.restore(1, np.asarray(eris.vvvv), nvir)
                else:
                    g_vvvv = eris.vvvv[:]
                g_cas[v,v,v,v] = einsum('IJKL,Ii,Jj,Kk,Ll->ijkl', g_vvvv, rv, rv, rv, rv)
            # Note that this will not work for 2D systems!:
            elif hasattr(eris, 'vvL') and eris.vvL is not None:
                if eris.vvL.ndim == 2:
                    naux = eris.vvL.shape[-1]
                    vvl = pyscf.lib.unpack_tril(eris.vvL, axis=0).reshape(nvir,nvir,naux)
                else:
                    vvl = eris.vvl
                vvl = einsum('IJQ,Ii,Jj->ijQ', vvl, rv, rv)
                g_cas[v,v,v,v] = einsum('ijQ,klQ->ijkl', vvl.conj(), vvl)
            else:
                raise RuntimeError("ERIs has not attribute 'vvvv' or 'vvL'.")
            self.log.timingv("Time to make CAS ERIs: %s", time_string(timer()-t0))
            return g_cas

        g_cas = make_cas_eris()
        # For the FCI, we need an effective one-electron Hamiltonian,
        # which contains Coulomb and exchange interaction to all frozen occupied orbitals
        # To calculate this, we would in principle need the whole-system 4c-integrals
        # Instead, we can start from the full system Fock matrix, which we already know
        # and subtract the parts NOT due to the frozen core density:
        # This Fock matrix does NOT contain exxdiv correction!
        #f_act = np.linalg.multi_dot((c_cas.T, eris.fock, c_cas))
        f_act = np.linalg.multi_dot((r.T, eris.fock, r))
        v_act = 2*einsum('iipq->pq', g_cas[o,o]) - einsum('iqpi->pq', g_cas[o,:,:,o])
        h_eff = f_act - v_act

        #fcisolver = pyscf.fci.direct_spin0.FCISolver(self.mol)
        fcisolver = pyscf.fci.direct_spin1.FCISolver(self.mol)

        if self.opts.tcc_spin is not None:
            fcisolver = pyscf.fci.addons.fix_spin_(fcisolver, ss=self.opts.tcc_spin)

        t0 = timer()
        e_fci, wf0 = fcisolver.kernel(h_eff, g_cas, ncas, nelec)
        self.log.timing("Time for FCI: %s", time_string(timer()-t0))
        if not fcisolver.converged:
            self.log.error("FCI not converged!")
        # Get C0,C1,and C2 from WF
        cisdvec = pyscf.ci.cisd.from_fcivec(wf0, ncas, nelec)
        c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, ncas, ncasocc)
        self.log.info("FCI weight on reference determinant: %.8g", abs(c0))
        if (c0 == 0):
            msg = "FCI wave function has no overlap with HF determinant."
            self.log.critical(msg)
            raise RuntimeError(msg)
        # Intermediate normalization
        c1 /= c0
        c2 /= c0
        t1_fci = c1
        t2_fci = c2 - einsum('ia,jb->ijab', c1, c1)

        def tailor_func(cc, t1, t2):
            # Rotate & project CC amplitudes to CAS
            t1_cc = einsum('IA,Ii,Aa->ia', t1, ro, rv)
            t2_cc = einsum('IJAB,Ii,Jj,Aa,Bb->ijab', t2, ro, ro, rv, rv)
            # Take difference wrt to FCI
            dt1 = (t1_fci - t1_cc)
            dt2 = (t2_fci - t2_cc)
            # Rotate back to CC space
            dt1 = einsum('ia,Ii,Aa->IA', dt1, ro, rv)
            dt2 = einsum('ijab,Ii,Jj,Aa,Bb->IJAB', dt2, ro, ro, rv, rv)
            # Add correction
            t1 += dt1
            t2 += dt2
            cc._norm_dt1 = np.linalg.norm(dt1)
            cc._norm_dt2 = np.linalg.norm(dt2)
            return t1, t2

        return tailor_func


    def make_sc_tailor_function(self, mode, correct_t1=True, correct_t2=True):
        """Build tailor function.

        This assumes orthogonal fragment spaces.
        """
        if mode not in (1, 2, 3):
            raise ValueError()
        ovlp = self.base.get_ovlp()     # AO overlap matrix
        c_occ = self.c_active_occ       # Occupied active orbitals of current cluster
        c_vir = self.c_active_vir       # Virtual  active orbitals of current cluster

        def tailor_func(cc, t1, t2):
            """Add external correction to T1 and T2 amplitudes."""

            # Add the correction to dt1 and dt2:
            if correct_t1:
                dt1 = np.zeros_like(t1)
            if correct_t2:
                dt2 = np.zeros_like(t2)

            # Loop over all *other* fragments/cluster X
            for fx in self.fragment.tailor_fragments:
                assert (fx is not self.fragment)
                cx_occ = fx.c_active_occ    # Occupied active orbitals of cluster X
                cx_vir = fx.c_active_vir    # Virtual  active orbitals of cluster X

                # Rotation & projections from cluster X active space to current fragment active space
                p_occ = np.linalg.multi_dot((cx_occ.T, ovlp, c_occ))
                p_vir = np.linalg.multi_dot((cx_vir.T, ovlp, c_vir))
                px = fx.get_fragment_projector(c_occ)   # this is C_occ^T . S . C_frag . C_frag^T . S . C_occ
                # Transform fragment X T-amplitudes to current active space and form difference
                if correct_t1:
                    tx1 = helper.transform_amplitude(fx.results.t1, p_occ, p_vir)   # ia,ix,ap->xp
                    dtx1 = (tx1 - t1)
                    dtx1 = np.dot(px, dtx1)
                    assert dtx1.shape == dt1.shape
                    dt1 += dtx1
                if correct_t2:
                    tx2 = helper.transform_amplitude(fx.results.t2, p_occ, p_vir)   # ijab,ix,jy,ap,bq->xypq
                    dtx2 = (tx2 - t2)
                    if mode == 1:
                        dtx2 = einsum('xi,yj,ijab->xyab', px, px, dtx2)
                    elif mode == 2:
                        py = self.fragment.get_fragment_projector(c_occ, inverse=True)
                        dtx2 = einsum('xi,yj,ijab->xyab', px, py, dtx2)
                    elif mode == 3:
                        dtx2 = einsum('xi,ijab->xjab', px, dtx2)
                    assert dtx2.shape == dt2.shape
                    dt2 += dtx2

                self.log.debugv("Tailoring %12s <- %12s: |dT1|= %.2e  |dT2|= %.2e", self.fragment, fx, np.linalg.norm(dtx1), np.linalg.norm(dtx2))

            # Store these norms in cc, to log their final value:
            if correct_t1:
                cc._norm_dt1 = np.linalg.norm(dt1)
            else:
                cc._norm_dt1 = 0.0
            if correct_t2:
                cc._norm_dt2 = np.linalg.norm(dt2)
            else:
                cc._norm_dt2 = 0.0
            # Add correction:
            if correct_t1:
                t1 = (t1 + dt1)
            if correct_t2:
                dt2 = (dt2 + dt2.transpose(1,0,3,2))/2
                t2 = (t2 + dt2)

            return t1, t2

        return tailor_func


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
class FCISolverResults(ClusterSolverResults):
    # CI coefficients
    c0: float = None
    c1: np.array = None
    c2: np.array = None


class FCISolver(ClusterSolver):
    """Not tested"""


    def kernel(self, init_guess=None):
        import pyscf.mcscf
        import pyscf.ci

        nelectron = sum(self.mo_occ[self.get_active_slice()])
        casci = pyscf.mcscf.CASCI(self.mf, self.nactive, nelectron)
        casci.canonicalization = False

        e_tot, e_cas, wf, *_ = casci.kernel(mo_coeff=self.mo_coeff)
        self.log.debug("FCI done. converged: %r", casci.converged)

        cisdvec = pyscf.ci.cisd.from_fcivec(wf, self.nactive, nelectron)
        nocc_active = nelectron // 2
        c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, self.nactive, nocc_active)
        # Intermediate normalization
        self.log.debug("Weight of reference determinant= %.8e", c0)
        c1 /= c0
        c2 /= c0
        self.c1 = c1
        self.c2 = c2

        self.converged = casci.converged
        self.e_corr = (e_tot - self.mf.e_tot)
        self.log.debug("E(full corr)= % 16.8f Ha", self.e_corr)

        ## Create fake CISD object
        #cisd = pyscf.ci.CISD(self.mf, mo_coeff=self.mo_coeff, mo_occ=self.mo_occ, frozen=self.get_frozen_indices())

        ## Get eris somewhere else?
        #t0 = timer()
        #eris = cisd.ao2mo()
        #self.log.debug("Time for integral transformation: %s", time_string(timer()-t0))


    #def run_fci(self):
    #    nocc_active = len(self.active_occ)
    #    casci = pyscf.mcscf.CASCI(self.mf, self.nactive, 2*nocc_active)
    #    solverobj = casci
    #    # Solver options
    #    casci.verbose = 10
    #    casci.canonicalization = False
    #    #casci.fix_spin_(ss=0)
    #    # TEST SPIN
    #    if solver == "FCI-spin0":
    #        casci.fcisolver = pyscf.fci.direct_spin0.FCISolver(self.mol)
    #    casci.fcisolver.conv_tol = 1e-9
    #    casci.fcisolver.threads = 1
    #    casci.fcisolver.max_cycle = 400
    #    #casci.fcisolver.level_shift = 5e-3

    #    if solver_options:
    #        spin = solver_options.pop("fix_spin", None)
    #        if spin is not None:
    #            self.log.debug("Setting fix_spin to %r", spin)
    #            casci.fix_spin_(ss=spin)

    #        for key, value in solver_options.items():
    #            self.log.debug("Setting solver attribute %s to value %r", key, value)
    #            setattr(casci.fcisolver, key, value)

    #    # The sorting of the orbitals above should already have placed the CAS in the correct position

    #    self.log.debug("Running FCI...")
    #    if self.nelectron_target is None:
    #        e_tot, e_cas, wf, *_ = casci.kernel(mo_coeff=mo_coeff)
    #    # Chemical potential loop
    #    else:

    #        S = self.mf.get_ovlp()
    #        px = self.get_local_projector(mo_coeff)
    #        b = np.linalg.multi_dot((S, self.C_local, self.C_local.T, S))

    #        t = np.linalg.multi_dot((S, mo_coeff, px))
    #        h1e = casci.get_hcore()
    #        h1e_func = casci.get_hcore

    #        cptmin = -4
    #        cptmax = 0
    #        #cptmin = -0.5
    #        #cptmax = +0.5

    #        e_tot = None
    #        wf = None

    #        def electron_error(chempot):
    #            nonlocal e_tot, wf

    #            #casci.get_hcore = lambda *args : h1e - chempot*b
    #            casci.get_hcore = lambda *args : h1e - chempot*(S-b)

    #            e_tot, e_cas, wf, *_ = casci.kernel(mo_coeff=mo_coeff, ci0=wf)
    #            #e_tot, e_cas, wf, *_ = casci.kernel(mo_coeff=mo_coeff)
    #            dm1xx = np.linalg.multi_dot((t.T, casci.make_rdm1(), t))
    #            nx = np.trace(dm1xx)
    #            nerr = (nx - self.nelectron_target)
    #            self.log.debug("chempot=%16.8g, electrons=%16.8g, error=%16.8g", chempot, nx, nerr)
    #            assert casci.converged

    #            if abs(nerr) < ntol:
    #                self.log.debug("Electron error |%e| below tolerance of %e", nerr, ntol)
    #                raise StopIteration

    #            return nerr

    #        try:
    #            scipy.optimize.brentq(electron_error, cptmin, cptmax)
    #        except StopIteration:
    #            pass

    #        # Reset hcore Hamiltonian
    #        casci.get_hcore = h1e_func

    #    #assert np.allclose(mo_coeff_casci, mo_coeff)
    #    #dma, dmb = casci.make_rdm1s()
    #    #self.log.debug("Alpha: %r", np.diag(dma))
    #    #self.log.debug("Beta: %r", np.diag(dmb))
    #    self.log.debug("FCI done. converged: %r", casci.converged)
    #    #self.log.debug("Shape of WF: %r", list(wf.shape))
    #    cisdvec = pyscf.ci.cisd.from_fcivec(wf, self.nactive, 2*nocc_active)
    #    C0, C1, C2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, self.nactive, nocc_active)
    #    # Intermediate normalization
    #    self.log.debug("Weight of reference determinant = %.8e", C0)
    #    renorm = 1/C0
    #    C1 *= renorm
    #    C2 *= renorm

    #    converged = casci.converged
    #    e_corr_full = self.energy_factor*(e_tot - self.mf.e_tot)

    #    # Create fake CISD object
    #    cisd = pyscf.ci.CISD(self.mf, mo_coeff=mo_coeff, mo_occ=mo_occ, frozen=frozen)

    #    if eris is None:
    #        t0 = MPI.Wtime()
    #        eris = cisd.ao2mo()
    #        self.log.debug("Time for integral transformation: %s", time_string(MPI.Wtime()-t0))

    #    pC1, pC2 = self.get_local_amplitudes(cisd, C1, C2)
    #    e_corr = self.get_local_energy(cisd, pC1, pC2, eris=eris)
