import dataclasses
from timeit import default_timer as timer

import numpy as np
import scipy
import scipy.optimize

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
from . import helper


def get_solver_class(solver):
    if solver.upper() in ('CCSD', 'CCSD(T)', 'TCCSD'):
        return CCSDSolver
    if solver.upper() == 'FCI':
        return FCISolver
    raise NotImplementedError("Unknown solver %s" % solver)


@dataclasses.dataclass
class ClusterSolverOptions(Options):
    make_rdm1: bool = NotSet
    make_rdm2: bool = NotSet
    sc_mode: int = NotSet
    # CCSD specific
    # EOM CCSD
    eom_ccsd: list = NotSet  # {'IP', 'EA', 'EE-S', 'EE-D', 'EE-SF'}
    eom_ccsd_nroots: int = NotSet
    # Active space methods
    c_cas_occ: np.array = None
    c_cas_vir: np.array = None
    # Tailored-CCSD
    tcc: bool = False
    tcc_fci_opts: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ClusterSolverResults:
    converged: bool = False
    e_corr: float = 0.0
    c_occ: np.array = None
    c_vir: np.array = None
    # Density matrix in MO representation:
    dm1: np.array = None
    dm2: np.array = None
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

    @property
    def c_active(self):
        return self.mo_coeff[:,self.get_active_slice()]

    def kernel_optimize_cpt(self, nelectron_target, *args, lower_bound=-1.0, upper_bound=1.0, tol=1e-8, **kwargs):
        get_hcore = self.base.mf.get_hcore
        h1e = get_hcore()
        # Fragment projector
        cs = np.dot(self.fragment.c_frag.T, self.base.get_ovlp())
        p_frag = np.dot(cs.T, cs)
        csc = np.dot(cs, self.c_active)

        self.opts.make_rdm1 = True
        results = None
        err = None

        def electron_err(cpt):
            nonlocal results, err
            h1e_cpt = h1e - cpt*p_frag
            # Not multi-threaded!
            self.base.mf.get_hcore = lambda *args : h1e_cpt
            results = self.kernel(*args, **kwargs)
            ne_frag = einsum('xi,ij,xj->', csc, results.dm1, csc)
            err = (ne_frag - nelectron_target)
            self.log.debugv("Electron number in fragment= %.8f  target=  %.8f  error= %.8f  chem. pot.=  %16.8f Ha", ne_frag, nelectron_target, err, cpt)
            return err

        for ndouble in range(5):
            try:
                cpt, res = scipy.optimize.brentq(electron_err, a=lower_bound, b=upper_bound, xtol=tol, full_output=True)
            except ValueError:
                if err < 0:
                    upper_bound *= 2
                else:
                    lower_bound *= 2
                self.log.debug("Bounds for chemical potential search too small. New bounds: [%f %f]", lower_bound, upper_bound)
                continue

            if res.converged:
                break
            else:
                errmsg = "Correct chemical potential not found."
                self.log.critical(errmsg)
                raise RuntimeError(errmsg)
        else:
            errmsg = "Could not find chemical potential within [%f %f]" % (lower_bound, upper_bound)
            self.log.critical(errmsg)
            raise RuntimeError(errmsg)

        self.log.info("Optimized chemical potential= % 16.8f Ha", cpt)

        # Restore
        self.base.mf.get_hcore = get_hcore
        return results


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
    # EE
    ee_s_energy: np.array = None
    ee_t_energy: np.array = None
    ee_sf_energy: np.array = None
    ee_s_coeff: np.array = None
    ee_t_coeff: np.array = None
    ee_sf_coeff: np.array = None



class CCSDSolver(ClusterSolver):

    def kernel(self, init_guess=None, eris=None, coupled_fragments=None, t_diagnostic=True):

        if coupled_fragments is None:
            coupled_fragments = self.fragment.coupled_fragments

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
            cc.tailor_func = self.make_cas_tcc_function(
                    c_cas_occ=self.opts.c_cas_occ, c_cas_vir=self.opts.c_cas_vir, eris=eris).__get__(cc)

        elif self.opts.sc_mode and self.base.iteration > 1:
            # __get__(cc) to bind the tailor function as a method,
            # rather than just a callable attribute
            self.log.info("Adding tailor function to CCSD.")
            cc.tailor_func = self.make_cross_fragment_tcc_function(mode=self.opts.sc_mode).__get__(cc)

        # This should include the SC mode?
        elif coupled_fragments and np.all([x.results is not None for x in coupled_fragments]):
            self.log.info("Adding tailor function to CCSD.")
            cc.tailor_func = self.make_cross_fragment_tcc_function(mode=(self.opts.sc_mode or 3),
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

        results = CCSDSolverResults(
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
            results.dm1 = cc.make_rdm1(with_frozen=False)
        if self.opts.make_rdm2:
            self.log.info("Making RDM2...")
            results.dm2 = cc.make_rdm2(with_frozen=False)

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


    def make_cas_tcc_function(self, c_cas_occ, c_cas_vir, eris):
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
        self.opts.tcc_fci_opts['max_cycle'] = self.opts.tcc_fci_opts.get('max_cycle', 1000)
        fix_spin = self.opts.tcc_fci_opts.pop('fix_spin', 0)
        if fix_spin not in (None, False):
            self.log.debugv("Fixing spin of FCIsolver to S^2= %r", fix_spin)
            fcisolver = pyscf.fci.addons.fix_spin_(fcisolver, ss=fix_spin)
        for key, val in self.opts.tcc_fci_opts.items():
            self.log.debugv("Setting FCIsolver attribute %s to %r", key, val)
            setattr(fcisolver, key, val)

        t0 = timer()
        e_fci, wf0 = fcisolver.kernel(h_eff, g_cas, ncas, nelec)
        self.log.timing("Time for FCI: %s", time_string(timer()-t0))
        if not fcisolver.converged:
            self.log.error("FCI not converged!")
        # Get C0,C1,and C2 from WF
        cisdvec = pyscf.ci.cisd.from_fcivec(wf0, ncas, nelec)
        c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, ncas, ncasocc)
        self.log.info("FCI weight on reference determinant: %.8g", abs(c0))
        if abs(c0) < 1e-4:
            self.log.warning("Weight on reference determinant small!")
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


    def make_cross_fragment_tcc_function(self, mode, coupled_fragments=None, correct_t1=True, correct_t2=True, symmetrize_t2=True):
        """Tailor current CCSD calculation with amplitudes of other fragments.

        This assumes orthogonal fragment spaces.

        Parameters
        ----------
        mode : int, optional
            Level of external correction of T2 amplitudes:
            1: Both occupied indices are projected to each other fragment X.
            2: Both occupied indices are projected to each other fragment X
               and combinations of other fragments X,Y.
            3: Only the first occupied indices is projected to each other fragment X.
        coupled_fragments : list, optional
            List of fragments, which are used for the external correction.
            Each fragment x must have the following attributes defined:
            `c_active_occ` : Active occupied MO orbitals of fragment x
            `c_active_vir` : Active virtual MO orbitals of fragment x
            `results.t1` :   T1 amplitudes of fragment x
            `results.t2` :   T2 amplitudes of fragment x

        Returns
        -------
        tailor_func : function(cc, t1, t2) -> t1, t2
            Tailoring function for CCSD.
        """
        if mode not in (1, 2, 3):
            raise ValueError()
        self.log.debugv("TCC mode= %d", mode)
        ovlp = self.base.get_ovlp()     # AO overlap matrix
        c_occ = self.c_active_occ       # Occupied active orbitals of current cluster
        c_vir = self.c_active_vir       # Virtual  active orbitals of current cluster

        if coupled_fragments is None:
            coupled_fragments = self.fragment.coupled_fragments

        def tailor_func(cc, t1, t2):
            """Add external correction to T1 and T2 amplitudes."""
            # Add the correction to dt1 and dt2:
            if correct_t1:  dt1 = np.zeros_like(t1)
            if correct_t2:  dt2 = np.zeros_like(t2)

            # Loop over all *other* fragments/cluster X
            for x in coupled_fragments:
                assert (x is not self.fragment)
                cx_occ = x.c_active_occ    # Occupied active orbitals of cluster X
                cx_vir = x.c_active_vir    # Virtual  active orbitals of cluster X

                # Rotation & projections from cluster X active space to current fragment active space
                p_occ = np.linalg.multi_dot((cx_occ.T, ovlp, c_occ))
                p_vir = np.linalg.multi_dot((cx_vir.T, ovlp, c_vir))
                px = x.get_fragment_projector(c_occ)   # this is C_occ^T . S . C_frag . C_frag^T . S . C_occ
                if x.results.t1 is None and x.results.c1 is not None:
                    self.log.debugv("Converting C-amplitudes of %s to T-amplitudes", x)
                    x.results.convert_amp_c_to_t()
                # Transform fragment X T-amplitudes to current active space and form difference
                if correct_t1:
                    tx1 = helper.transform_amplitude(x.results.t1, p_occ, p_vir)   # ia,ix,ap->xp
                    dtx1 = (tx1 - t1)
                    dtx1 = np.dot(px, dtx1)
                    assert dtx1.shape == dt1.shape
                    dt1 += dtx1
                if correct_t2:
                    tx2 = helper.transform_amplitude(x.results.t2, p_occ, p_vir)   # ijab,ix,jy,ap,bq->xypq
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
                self.log.debugv("Tailoring %12s <- %12s: |dT1|= %.2e  |dT2|= %.2e", self.fragment, x, np.linalg.norm(dtx1), np.linalg.norm(dtx2))

            # Store these norms in cc, to log their final value:
            cc._norm_dt1 = np.linalg.norm(dt1) if correct_t1 else 0.0
            cc._norm_dt2 = np.linalg.norm(dt2) if correct_t2 else 0.0
            # Add correction:
            if correct_t1:
                t1 = (t1 + dt1)
            if correct_t2:
                if symmetrize_t2:
                    self.log.debugv("T2 symmetry error: %e", np.linalg.norm(dt2 - dt2.transpose(1,0,3,2))/2)
                    dt2 = (dt2 + dt2.transpose(1,0,3,2))/2
                t2 = (t2 + dt2)
            return t1, t2

        return tailor_func


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
        if self.mf._eri is not None:
            class ERIs:
                pass
            eris = ERIs()
            c_act = self.mo_coeff[:,self.get_active_slice()]
            eris.fock = np.linalg.multi_dot((c_act.T, self.base.get_fock(), c_act))
            g = pyscf.ao2mo.full(self.mf._eri, c_act)
            o = np.s_[:nocc]
            v = np.s_[nocc:]
            eris.ovvo = pyscf.ao2mo.restore(1, g, self.nactive)[o,v,v,o]
        else:
            # TODO
            pass

        results = CISolverResults(
                converged=casci.converged, e_corr=e_corr, c_occ=self.c_active_occ, c_vir=self.c_active_vir, eris=eris,
                c0=c0, c1=c1, c2=c2)

        if self.opts.make_rdm2:
            results.dm1, results.dm2 = casci.fcisolver.make_rdm12(wf, self.nactive, nelec)
        elif self.opts.make_rdm1:
            results.dm1 = casci.fcisolver.make_rdm1(wf, self.nactive, nelec)

        return results


    #def kernel2(self, init_guess=None, eris=None):
    #    """TODO"""

    #    c_act = self.mo_coeff[:,self.get_active_slice()]
    #    if eris is None:
    #        # Temporary implementation
    #        import pyscf.ao2mo
    #        t0 = timer()
    #        eris = pyscf.ao2mo.general(self.mf._eri, c_act)
    #        self.log.timing("Time for AO->MO of (ij|kl):  %s", time_string(timer()-t0))


    #    f_act = np.linalg.multi_dot((r.T, eris.fock, r))
    #    v_act = 2*einsum('iipq->pq', eris[o,o]) - einsum('iqpi->pq', g_cas[o,:,:,o])
    #    h_eff = f_act - v_act


    #    fcisolver = pyscf.fci.direct_spin1.FCISolver(self.mol)
    #    t0 = timer()
    #    e_fci, wf0 = fcisolver.kernel(h_eff, g_cas, ncas, nelec)
    #    if not fcisolver.converged:
    #        self.log.error("FCI not converged!")
    #    self.log.timing("Time for FCI: %s", time_string(timer()-t0))



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
