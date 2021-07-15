# Standard libaries
import os
import os.path
from collections import OrderedDict
import functools
from datetime import datetime
from timeit import default_timer as timer
import dataclasses
import copy
import gc

# External libaries
import numpy as np
import scipy
import scipy.linalg

# Internal libaries
import pyscf
#import pyscf.pbc
from pyscf.pbc.tools import cubegen

# Local modules
from vayesta.core.util import *
from vayesta.core import QEmbeddingFragment
# We might want to move the useful things from here into core, since they seem pretty general.
from vayesta.ewf import helper, psubspace


from . import dmet
from .solver import get_solver_class
from vayesta.ewf import mp2_bath


@dataclasses.dataclass
class DMETFragmentOptions(Options):
    """Attributes set to `NotSet` inherit their value from the parent EWF object."""
    # Options also present in `base`:
    dmet_threshold: float = NotSet
    make_rdm1: bool = True
    eom_ccsd: bool = NotSet
    plot_orbitals: str = NotSet
    bsse_correction: bool = NotSet
    bsse_rmax: float = NotSet
    energy_partitioning: str = NotSet
    pop_analysis: str = NotSet
    sc_mode: int = NotSet
    # Additional fragment specific options:
    bno_threshold_factor: float = 1.0
    # CAS methods
    c_cas_occ: np.ndarray = None
    c_cas_vir: np.ndarray = None


@dataclasses.dataclass
class DMETFragmentResults:
    fid: int = None
    bno_threshold: float = None
    n_active: int = None
    converged: bool = None
    e_corr: float = None
    ip_energy: np.ndarray = None
    ea_energy: np.ndarray = None
    c0: float = None
    c1: np.ndarray = None
    c2: np.ndarray = None
    t1: np.ndarray = None
    t2: np.ndarray = None
    l1: np.ndarray = None
    l2: np.ndarray = None
    eris: 'typing.Any' = None
    # For DM1:
    g1: np.ndarray = None
    dm1: np.ndarray = None


class DMETFragmentExit(Exception):
    pass


class DMETFragment(QEmbeddingFragment):
    def __init__(self, base, fid, name, c_frag, c_env, fragment_type, sym_factor=1, atoms=None, aos=None, log=None,
            solver=None, options=None, **kwargs):
        """
        Parameters
        ----------
        base : EWF
            Base EWF object.
        fid : int
            Unique ID of fragment.
        name :
            Name of fragment.
        """

        super().__init__(base, fid, name, c_frag, c_env, fragment_type, sym_factor=sym_factor, atoms=atoms, aos=aos, log=log)

        if options is None:
            options = DMETFragmentOptions(**kwargs)
        else:
            options = options.replace(kwargs)
        options = options.replace(self.base.opts, select=NotSet)
        if options.pop_analysis:
            options.make_rdm1 = True
        self.opts = options
        for key, val in self.opts.items():
            self.log.infov('  > %-24s %r', key + ':', val)

        if solver is None:
            solver = self.base.solver
        if solver not in dmet.VALID_SOLVERS:
            raise ValueError("Unknown solver: %s" % solver)
        self.solver = solver
        self.log.infov('  > %-24s %r', 'Solver:', self.solver)

        # --- These attributes will be set after calling `make_bath`:
        # DMET-cluster (fragment + DMET bath) orbital coefficients
        self.c_cluster_occ = None
        self.c_cluster_vir = None
        # BNO orbital coefficients
        self.c_no_occ = None
        self.c_no_vir = None
        # BNO orbital occupation numbers
        self.n_no_occ = None
        self.n_no_vir = None

        # --- Attributes which will be overwritten for each BNO threshold:
        # Active orbitals
        self.c_active_occ = None
        self.c_active_vir = None

        # For self-consistent mode
        self.solver_results = None

        self.results = None

    @property
    def e_corr(self):
        """Best guess for correlation energy, using the lowest BNO threshold."""
        idx = np.argmin(self.bno_threshold)
        return self.e_corrs[idx]


    def make_bath(self, rdm):
        """Make DMET and MP2 bath natural orbitals."""
        t0_bath = t0 = timer()
        self.log.info("Making DMET Bath")
        self.log.info("****************")
        self.log.changeIndentLevel(1)
        c_dmet, c_env_occ, c_env_vir = self.make_dmet_bath(self.c_env, tol=self.opts.dmet_threshold, rdm = rdm)
        # DMET bath analysis
        self.log.info("DMET bath character:")
        for i in range(c_dmet.shape[-1]):
            ovlp = einsum('a,b,ba->a', c_dmet[:,i], c_dmet[:,i], self.base.get_ovlp())
            sort = np.argsort(-ovlp)
            ovlp = ovlp[sort]
            #n = np.amin((np.count_nonzero(ovlp >= 0.2), 5))
            n = np.amin((len(ovlp), 6))
            labels = np.asarray(self.mol.ao_labels())[sort][:n]
            lines = [("%s= %.5f" % (labels[i].strip(), ovlp[i])) for i in range(n)]
            self.log.info("  > %2d:  %s", i+1, '  '.join(lines))

        self.log.timing("Time for DMET bath:  %s", time_string(timer()-t0))
        self.log.changeIndentLevel(-1)

        # Add fragment and DMET orbitals for cube file plots
        if self.opts.plot_orbitals:
            if self.boundary_cond == 'open':
                raise NotImplementedError()
            os.makedirs(self.base.opts.plot_orbitals_dir, exist_ok=True)
            name = "%s.cube" % os.path.join(self.base.opts.plot_orbitals_dir, self.name)
            cubefile = cubegen.CubeFile(self.mol, filename=name, **self.base.opts.plot_orbitals_kwargs)
            self.log.debug("Adding fragment orbitals to cube file.")
            cubefile.add_orbital(self.c_frag.copy())
            if (self.opts.plot_orbitals is True) or ('dmet' in str(self.opts.plot_orbitals).lower()):
                self.log.debug("Adding DMET orbitals to cube file.")
                cubefile.add_orbital(c_dmet.copy(), dset_idx=1001)
            if str(self.opts.plot_orbitals).lower() in ('fragment-exit', 'dmet-exit'):
                cubefile.write()
                raise DMETFragmentExit()

        # Add additional orbitals to cluster [optional]
        #c_dmet, c_env_occ, c_env_vir = self.additional_bath_for_cluster(c_dmet, c_env_occ, c_env_vir)

        # Diagonalize cluster DM to separate cluster occupied and virtual
        self.c_cluster_occ, self.c_cluster_vir = self.diagonalize_cluster_dm(self.c_frag, c_dmet, tol=2*self.opts.dmet_threshold)
        self.log.info("Cluster orbitals:  n(occ)= %3d  n(vir)= %3d", self.c_cluster_occ.shape[-1], self.c_cluster_vir.shape[-1])


        if c_env_occ.shape[-1] > 0:
            self.log.info("Making Occupied Bath NOs")
            self.log.info("------------------------")
            t0 = timer()
            self.log.changeIndentLevel(1)
            self.c_no_occ, self.n_no_occ = mp2_bath.make_mp2_bno(
                    self, "occ", self.c_cluster_occ, self.c_cluster_vir, c_env_occ, c_env_vir)
            self.log.timing("Time for occupied BNOs:  %s", time_string(timer()-t0))
            if len(self.n_no_occ) > 0:
                self.log.info("Occupied Bath NO Histogram")
                self.log.info("--------------------------")
                for line in helper.plot_histogram(self.n_no_occ):
                    self.log.info(line)
            self.log.changeIndentLevel(-1)
        else:
            self.c_no_occ = c_env_occ
            self.n_no_occ = np.zeros((0,))

        if c_env_vir.shape[-1] > 0:
            self.log.info("Making Virtual Bath NOs")
            self.log.info("-----------------------")
            t0 = timer()
            self.log.changeIndentLevel(1)
            self.c_no_vir, self.n_no_vir = mp2_bath.make_mp2_bno(
                    self, "vir", self.c_cluster_occ, self.c_cluster_vir, c_env_occ, c_env_vir)
            self.log.timing("Time for virtual BNOs:   %s", time_string(timer()-t0))
            if len(self.n_no_vir) > 0:
                self.log.info("Virtual Bath NO Histogram")
                self.log.info("-------------------------")
                for line in helper.plot_histogram(self.n_no_vir):
                    self.log.info(line)
            self.log.changeIndentLevel(-1)
        else:
            self.c_no_vir = c_env_vir
            self.n_no_vir = np.zeros((0,))

        # Plot orbitals
        if self.opts.plot_orbitals:
            # Save state of cubefile, in case a replot of the same data is required later:
            cubefile.save_state("%s.pkl" % cubefile.filename)
            cubefile.write()
        self.log.timing("Time for bath:  %s", time_string(timer()-t0_bath))

    def set_cas(self, iaos=None, c_occ=None, c_vir=None):
        if iaos is not None:
            # Convert to index array
            iaos = self.base.get_ao_indices(iaos, 'IAO')
            c_iao = self.base.iao_coeff[:,iaos]
            rest_iaos = np.setdiff1d(range(self.base.iao_coeff.shape[-1]), iaos)
            # Combine remaining IAOs and rest virtual space (`iao_rest_coeff`)
            c_env = np.hstack((self.base.iao_coeff[:,rest_iaos], self.base.iao_rest_coeff))
            c_dmet = self.make_dmet_bath(c_env, tol=self.opts.dmet_threshold)[0]

            c_iao_occ, c_iao_vir = self.diagonalize_cluster_dm(c_iao, c_dmet, tol=2*self.opts.dmet_threshold)
        else:
            c_iao_occ = c_iao_vir = None

        def combine(c1, c2):
            if c1 is not None and c2 is not None:
                return np.hstack((c1, c2))
            if c1 is not None:
                return c1
            if c2 is not None:
                return c2
            raise ValueError()

        c_cas_occ = combine(c_occ, c_iao_occ)
        c_cas_vir = combine(c_vir, c_iao_vir)
        self.opts.c_cas_occ = c_cas_occ
        self.opts.c_cas_vir = c_cas_vir
        return c_cas_occ, c_cas_vir


    def kernel(self, rdm, bno_threshold, solver=None, init_guess=None, eris=None):
        """Run solver for a single BNO threshold.

        Parameters
        ----------
        bno_threshold : float
            Bath natural orbital (BNO) thresholds.
        solver : {'MP2', 'CISD', 'CCSD', 'CCSD(T)', 'FCI'}, optional
            Correlated solver.

        Returns
        -------
        results : DMETFragmentResults
        """
        solver = solver or self.solver
        # We always want to regenerate our bath orbitals.
        #if self.c_cluster_occ is None:
        self.make_bath(rdm)

        #self.e_delta_mp2 = e_delta_occ + e_delta_vir
        #self.log.debug("MP2 correction = %.8g", self.e_delta_mp2)

        assert (self.c_no_occ is not None)
        assert (self.c_no_vir is not None)

        bno_thr = self.opts.bno_threshold_factor * bno_threshold
        self.log.info("Occupied BNOs:")
        c_nbo_occ, c_frozen_occ = self.apply_bno_threshold(self.c_no_occ, self.n_no_occ, bno_thr)
        self.log.info("Virtual BNOs:")
        c_nbo_vir, c_frozen_vir = self.apply_bno_threshold(self.c_no_vir, self.n_no_vir, bno_thr)

        # Canonicalize orbitals
        c_active_occ = self.canonicalize_mo(self.c_cluster_occ, c_nbo_occ)[0]
        c_active_vir = self.canonicalize_mo(self.c_cluster_vir, c_nbo_vir)[0]
        # Do not overwrite self.c_active_occ/vir yet - we still need the previous coefficients
        # to generate an intial guess

        # Combine, important to keep occupied orbitals first!
        # Put frozen (occenv, virenv) orbitals to the front and back
        # and active orbitals (occact, viract) in the middle
        c_occ = np.hstack((c_frozen_occ, c_active_occ))
        c_vir = np.hstack((c_active_vir, c_frozen_vir))
        nocc, nvir = c_occ.shape[-1], c_vir.shape[-1]
        mo_coeff = np.hstack((c_occ, c_vir))

        # Check occupations
        n_occ = self.get_mo_occupation(c_occ)
        if not np.allclose(n_occ, 2, atol=2*self.opts.dmet_threshold):
            raise RuntimeError("Incorrect occupation of occupied orbitals:\n%r" % n_occ)
        n_vir = self.get_mo_occupation(c_vir)
        if not np.allclose(n_vir, 0, atol=2*self.opts.dmet_threshold):
            raise RuntimeError("Incorrect occupation of virtual orbitals:\n%r" % n_vir)
        mo_occ = np.asarray(nocc*[2] + nvir*[0])

        nocc_frozen = c_frozen_occ.shape[-1]
        nvir_frozen = c_frozen_vir.shape[-1]
        nfrozen = nocc_frozen + nvir_frozen
        nactive = c_active_occ.shape[-1] + c_active_vir.shape[-1]

        self.log.info("Orbitals for %s", self)
        self.log.info("*************" + len(str(self))*"*")
        self.log.info("  > Active:   n(occ)= %5d  n(vir)= %5d  n(tot)= %5d", c_active_occ.shape[-1], c_active_vir.shape[-1], nactive)
        self.log.info("  > Frozen:   n(occ)= %5d  n(vir)= %5d  n(tot)= %5d", nocc_frozen, nvir_frozen, nfrozen)
        self.log.info("  > Total:    n(occ)= %5d  n(vir)= %5d  n(tot)= %5d", c_occ.shape[-1], c_vir.shape[-1], mo_coeff.shape[-1])

        ## --- Do nothing if solver is not set
        #if not solver:
        #    self.log.info("Solver set to None. Skipping calculation.")
        #    self.converged = True
        #    return 0, nactive, None, None

        # --- Project initial guess and integrals from previous cluster calculation with smaller eta:
        # Use initial guess from previous calculations
        # For self-consistent calculations, we can restart calculation:
        if init_guess is None:
            if self.base.opts.sc_mode and self.base.iteration > 1:
                self.log.debugv("Restarting using T1,T2 from previous iteration")
                init_guess = {'t1' : self.results.t1, 't2' : self.results.t2}
            #elif self.base.opts.project_init_guess and init_guess is not None:
            #    # Projectors for occupied and virtual orbitals
            #    p_occ = np.linalg.multi_dot((self.c_active_occ.T, self.base.get_ovlp(), c_active_occ))
            #    p_vir = np.linalg.multi_dot((self.c_active_vir.T, self.base.get_ovlp(), c_active_vir))
            #    t1, t2 = init_guess.pop('t1'), init_guess.pop('t2')
            #    t1, t2 = helper.transform_amplitudes(t1, t2, p_occ, p_vir)
            #    init_guess['t1'] = t1
            #    init_guess['t2'] = t2
            elif self.base.opts.project_init_guess and self.results is not None:
                self.log.debugv("Restarting using projected previous T1,T2")
                # Projectors for occupied and virtual orbitals
                p_occ = np.linalg.multi_dot((self.c_active_occ.T, self.base.get_ovlp(), c_active_occ))
                p_vir = np.linalg.multi_dot((self.c_active_vir.T, self.base.get_ovlp(), c_active_vir))
                #t1, t2 = init_guess.pop('t1'), init_guess.pop('t2')
                t1, t2 = helper.transform_amplitudes(self.results.t1, self.results.t2, p_occ, p_vir)
                init_guess = {'t1' : t1, 't2' : t2}


        # For self-consistent calculations, we can reuse ERIs:
        if eris is None:
            if self.base.opts.sc_mode and self.base.iteration > 1:
                self.log.debugv("Reusing ERIs from previous iteration")
                eris = self.results.eris
            # If superspace ERIs were calculated before, they can be transformed and used again:
            elif self.base.opts.project_eris and self.results is not None:
                t0 = timer()
                self.log.debugv("Projecting previous ERIs onto subspace")
                eris = psubspace.project_eris(self.results.eris, c_active_occ, c_active_vir, ovlp=self.base.get_ovlp())
                self.log.timingv("Time to project ERIs:  %s", time_string(timer()-t0))

        # We can now overwrite the orbitals from last BNO run:
        self.c_active_occ = c_active_occ
        self.c_active_vir = c_active_vir

        # Create solver object
        t0 = timer()
        solver_opts = {}
        solver_opts['make_rdm1'] = self.opts.make_rdm1
        if solver.upper() == 'TCCSD':
            solver_opts['tcc'] = True
            # Set CAS orbitals
            if self.opts.c_cas_occ is None:
                self.log.warning("Occupied CAS orbitals not set. Setting to occupied DMET cluster orbitals.")
                self.opts.c_cas_occ = self.c_cluster_occ
            if self.opts.c_cas_vir is None:
                self.log.warning("Virtual CAS orbitals not set. Setting to virtual DMET cluster orbitals.")
                self.opts.c_cas_vir = self.c_cluster_vir
            solver_opts['c_cas_occ'] = self.opts.c_cas_occ
            solver_opts['c_cas_vir'] = self.opts.c_cas_vir
            # Temporary:
            solver_opts['tcc_spin'] = 0

        cluster_solver_cls = get_solver_class(solver)
        cluster_solver = cluster_solver_cls(self, mo_coeff, mo_occ, nocc_frozen=nocc_frozen, nvir_frozen=nvir_frozen, **solver_opts)
        solver_results = cluster_solver.kernel(init_guess=init_guess, eris=eris)
        self.log.timing("Time for %s solver:  %s", solver, time_string(timer()-t0))

        # Get projected amplitudes ('p1', 'p2')
        if hasattr(solver_results, 't1'):
            c1 = solver_results.t1
            c2 = solver_results.t2 + einsum('ia,jb->ijab', c1, c1)
        elif hasattr(solver_results, 'c1'):
            self.log.info("Weight of reference determinant= %.8g", abs(solver_results.c0))
            c1 = solver_results.c1 / solver_results.c0
            c2 = solver_results.c2 / solver_results.c0
        p1 = self.project_amplitude_to_fragment(c1, c_active_occ, c_active_vir)
        p2 = self.project_amplitude_to_fragment(c2, c_active_occ, c_active_vir)

        e_corr = self.get_fragment_energy(p1, p2, eris=solver_results.eris)
        self.log.info("BNO threshold= %.1e :  E(corr)= %+14.8f Ha", bno_threshold, e_corr)
        # Add frozen states and transform to AO
        dm1 = np.zeros(2 * [self.base.nao])
        nocc = np.count_nonzero(self.mf.mo_occ > 0)
        dm1[np.diag_indices(nocc)] = 2
        a = cluster_solver.get_active_slice()
        dm1[a, a] = solver_results.dm1
        # Population analysis
        if self.opts.pop_analysis:
            try:
                if isinstance(self.base.opts.pop_analysis, str):
                    filename = self.base.opts.pop_analysis.rsplit('.', 1)
                    if len(filename) > 1:
                        filename, ext = filename
                    else:
                        ext = 'txt'
                    filename = '%s-%s.%s' % (filename, self.id_name, ext)
                else:
                    filename = None
                self.base.pop_analysis(dm1, mo_coeff=mo_coeff, filename=filename)
            except Exception as e:
                self.log.error("Exception in population analysis: %s", e)

        results = DMETFragmentResults(
                fid=self.id,
                bno_threshold=bno_threshold,
                n_active=nactive,
                converged=solver_results.converged,
                e_corr=e_corr,
                dm1 = dm1)
        # EOM analysis
        if self.opts.eom_ccsd in (True, 'IP'):
            results.ip_energy, _ = self.eom_analysis(cluster_solver, 'IP')
        if self.opts.eom_ccsd in (True, 'EA'):
            results.ea_energy, _ = self.eom_analysis(cluster_solver, 'EA')

        # Keep Amplitudes [optional]
        if self.base.opts.project_init_guess or self.opts.sc_mode:
            if hasattr(solver_results, 't1'):
                results.t1 = solver_results.t1
                results.t2 = solver_results.t2
            if hasattr(solver_results, 'c1'):
                results.c0 = solver_results.c0
                results.c1 = solver_results.c1
                results.c2 = solver_results.c2
        # Keep Lambda-Amplitudes
        if results.l1 is not None:
            results.l1 = solver_results.l1
            results.l2 = solver_results.l2
        # Keep ERIs [optional]
        if self.base.opts.project_eris or self.opts.sc_mode:
            results.eris = solver_results.eris

        self.results = results

        # Force GC to free memory
        m0 = get_used_memory()
        del cluster_solver, solver_results
        ndel = gc.collect()
        self.log.debugv("GC deleted %d objects and freed %.3f MB of memory", ndel, (get_used_memory()-m0)/1e6)

        return results


    def apply_bno_threshold(self, c_no, n_no, bno_thr):
        """Split natural orbitals (NO) into bath and rest."""
        n_bno = np.count_nonzero(n_no >= bno_thr)
        # Logging
        fmt = "  > %4s: N= %4d  max= % 9.3g  min= % 9.3g  sum= % 9.3g ( %7.3f %%)"
        def log(name, n_part):
            if len(n_part) > 0:
                with np.errstate(invalid='ignore'): # supress 0/0=nan warning
                    self.log.info(fmt, name, len(n_part), max(n_part), min(n_part), np.sum(n_part),
                            100*np.sum(n_part)/np.sum(n_no))
            else:
                self.log.info(fmt[:fmt.index('max')], name, 0)
        log("Bath", n_no[:n_bno])
        log("Rest", n_no[n_bno:])

        c_bno, c_rest = np.hsplit(c_no, [n_bno])
        return c_bno, c_rest

    def project_amplitudes_to_fragment(self, cm, c1, c2, **kwargs):
        """Wrapper for project_amplitude_to_fragment, where the mo coefficients are extracted from a MP2 or CC object."""

        act = cm.get_frozen_mask()
        occ = cm.mo_occ[act] > 0
        vir = cm.mo_occ[act] == 0
        c = cm.mo_coeff[:,act]
        c_occ = c[:,occ]
        c_vir = c[:,vir]

        p1 = p2 = None
        if c1 is not None:
            p1 = self.project_amplitude_to_fragment(c1, c_occ, c_vir, **kwargs)
        if c2 is not None:
            p2 = self.project_amplitude_to_fragment(c2, c_occ, c_vir, **kwargs)
        return p1, p2


    def project_amplitude_to_fragment(self, c, c_occ=None, c_vir=None, partitioning=None, symmetrize=False):
        """Get local contribution of amplitudes."""

        if np.ndim(c) not in (2, 4):
            raise NotImplementedError()
        if partitioning is None:
            part = self.opts.energy_partitioning
        else:
            part = partitioning
        if part not in ('first-occ', 'first-vir', 'democratic'):
            raise ValueError("Unknown partitioning of amplitudes: %s" % part)

        # Projectors into fragment occupied and virtual space
        if part in ("first-occ", "democratic"):
            assert c_occ is not None
            fo = self.get_fragment_projector(c_occ)
        if part in ("first-vir", "democratic"):
            assert c_vir is not None
            fv = self.get_fragment_projector(c_vir)
        # Inverse projectors needed
        if part == "democratic":
            ro = np.eye(fo.shape[-1]) - fo
            rv = np.eye(fv.shape[-1]) - fv

        if np.ndim(c) == 2:
            if part == "first-occ":
                p = einsum("xi,ia->xa", fo, c)
            elif part == "first-vir":
                p = einsum("ia,xa->ix", c, fv)
            elif part == "democratic":
                p = einsum("xi,ia,ya->xy", fo, c, fv)
                p += einsum("xi,ia,ya->xy", fo, c, rv) / 2.0
                p += einsum("xi,ia,ya->xy", ro, c, fv) / 2.0
            return p

        # ndim == 4:

        if part == "first-occ":
            p = einsum("xi,ijab->xjab", fo, c)
        elif part == "first-vir":
            p = einsum("ijab,xa->ijxb", c, fv)
        elif part == "democratic":

            def project(p1, p2, p3, p4):
                p = einsum("xi,yj,ijab,za,wb->xyzw", p1, p2, c, p3, p4)
                return p

            # Factors of 2 due to ij,ab <-> ji,ba symmetry
            # Denominators 1/N due to element being shared between N clusters

            # Quadruple F
            # ===========
            # This is fully included
            p = project(fo, fo, fv, fv)
            # Triple F
            # ========
            # This is fully included
            p += 2*project(fo, fo, fv, rv)
            p += 2*project(fo, ro, fv, fv)
            # Double F
            # ========
            # P(FFrr) [This wrongly includes: 1x P(FFaa), instead of 0.5x - correction below]
            p +=   project(fo, fo, rv, rv)
            p += 2*project(fo, ro, fv, rv)
            p += 2*project(fo, ro, rv, fv)
            p +=   project(ro, ro, fv, fv)
            # Single F
            # ========
            # P(Frrr) [This wrongly includes: P(Faar) (where r could be a) - correction below]
            p += 2*project(fo, ro, rv, rv) / 4.0
            p += 2*project(ro, ro, fv, rv) / 4.0

            # Corrections
            # ===========
            # Loop over all other clusters x
            for x in self.loop_fragments(exclude_self=True):

                xo = x.get_fragment_projector(c_occ)
                xv = x.get_fragment_projector(c_vir)

                # Double correction
                # -----------------
                # Correct for wrong inclusion of P(FFaa)
                # The case P(FFaa) was included with prefactor of 1 instead of 1/2
                # We thus need to only correct by "-1/2"
                p -=   project(fo, fo, xv, xv) / 2.0
                p -= 2*project(fo, xo, fv, xv) / 2.0
                p -= 2*project(fo, xo, xv, fv) / 2.0
                p -=   project(xo, xo, fv, fv) / 2.0

                # Single correction
                # -----------------
                # Correct for wrong inclusion of P(Faar)
                # This corrects the case P(Faab) but overcorrects P(Faaa)!
                p -= 2*project(fo, xo, xv, rv) / 4.0
                p -= 2*project(fo, xo, rv, xv) / 4.0 # If r == x this is the same as above -> overcorrection
                p -= 2*project(fo, ro, xv, xv) / 4.0 # overcorrection
                p -= 2*project(xo, xo, fv, rv) / 4.0
                p -= 2*project(xo, ro, fv, xv) / 4.0 # overcorrection
                p -= 2*project(ro, xo, fv, xv) / 4.0 # overcorrection

                # Correct overcorrection
                # The additional factor of 2 comes from how often the term was wrongly included above
                p += 2*2*project(fo, xo, xv, xv) / 4.0
                p += 2*2*project(xo, xo, fv, xv) / 4.0

        # Note that the energy should be invariant to symmetrization
        if symmetrize:
            p = (p + p.transpose(1,0,3,2)) / 2

        return p


    def get_fragment_energy(self, p1, p2, eris):
        """
        Parameters
        ----------
        p1 : (nOcc, nVir) array
            Locally projected C1 amplitudes.
        p2 : (nOcc, nOcc, nVir, nVir) array
            Locally projected C2 amplitudes.
        eris :
            PySCF eris object as returned by cm.ao2mo()

        Returns
        -------
        e_frag : float
            Fragment energy contribution.
        """
        nocc, nvir = p2.shape[1:3]
        # MP2
        if p1 is None:
            e1 = 0
        # CC
        else:
            occ = np.s_[:nocc]
            vir = np.s_[nocc:]
            f = eris.fock[occ][:,vir]
            e1 = 2*np.sum(f * p1)

        if hasattr(eris, "ovvo"):
            eris_ovvo = eris.ovvo
        # MP2 only has eris.ovov - for real integrals we transpose
        else:
            eris_ovvo = eris.ovov[:].reshape(nocc,nvir,nocc,nvir).transpose(0, 1, 3, 2).conj()
        e2 = (2*einsum('ijab,iabj', p2, eris_ovvo)
              - einsum('ijab,jabi', p2, eris_ovvo))

        self.log.info("Energy components: E[C1]= % 16.8f Ha, E[C2]= % 16.8f Ha", e1, e2)
        if e1 > 1e-4 and 10*e1 > e2:
            self.log.warning("WARNING: Large E[C1] component!")

        e_frag = self.sym_factor * (e1 + e2)
        return e_frag

