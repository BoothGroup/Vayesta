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
import pyscf.pbc
from pyscf.pbc.tools import cubegen

# Local modules
from vayesta.core.util import *
from vayesta.core import QEmbeddingFragment
from vayesta.solver import get_solver_class

from . import ewf
from .mp2_bath import make_mp2_bno
from . import helper
from . import psubspace


class EWFFragment(QEmbeddingFragment):

    @dataclasses.dataclass
    class Options(QEmbeddingFragment.Options):
        """Attributes set to `NotSet` inherit their value from the parent EWF object."""
        # Options also present in `base`:
        dmet_threshold: float = NotSet
        make_rdm1: bool = NotSet
        make_rdm2: bool = NotSet
        eom_ccsd: list = NotSet
        eom_ccsd_nroots: int = NotSet
        bsse_correction: bool = NotSet
        bsse_rmax: float = NotSet
        energy_factor: float = 1.0
        #energy_partitioning: str = NotSet
        pop_analysis: str = NotSet
        sc_mode: int = NotSet
        nelectron_target: int = NotSet                  # If set, adjust bath chemical potential until electron number in fragment equals nelectron_target
        # Bath type
        bath_type: str = NotSet
        # Additional fragment specific options:
        bno_threshold_factor: float = 1.0
        # CAS methods
        c_cas_occ: np.ndarray = None
        c_cas_vir: np.ndarray = None
        # --- Orbital plots
        plot_orbitals: list = NotSet
        plot_orbitals_exit: bool = NotSet            # Exit immediately after all orbital plots have been generated
        plot_orbitals_dir: str = NotSet
        plot_orbitals_kwargs: dict = NotSet
        plot_orbitals_gridsize: tuple = NotSet
        # --- Solver options
        tcc_fci_opts: dict = dataclasses.field(default_factory=dict)

    @dataclasses.dataclass
    class Results(QEmbeddingFragment.Results):
        bno_threshold: float = None
        n_active: int = None
        ip_energy: np.ndarray = None
        ea_energy: np.ndarray = None
        eris: 'typing.Any' = None
        #e1b: float = None
        #e2b_conn: float = None
        #e2b_disc: float = None

    def __init__(self, *args, solver=None, **kwargs):

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

        super().__init__(*args, **kwargs)

        if self.opts.pop_analysis:
            self.opts.make_rdm1 = True

        # Default options:
        defaults = self.Options().replace(self.base.Options(), select=NotSet)
        for key, val in self.opts.items():
            if val != getattr(defaults, key):
                self.log.info('  > %-24s %3s %r', key + ':', '(*)', val)
            else:
                self.log.debugv('  > %-24s %3s %r', key + ':', '', val)

        if solver is None:
            solver = self.base.solver
        if solver not in ewf.VALID_SOLVERS:
            raise ValueError("Unknown solver: %s" % solver)
        self.solver = solver
        self.log.infov('  > %-24s %3s %r', 'Solver:', '', self.solver)


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

        # For self-consistent mode
        self.solver_results = None

        # For orbital plotting
        self.cubefile = None

        # TEMPORARY:
        self._c_frozen_occ = None
        self._c_frozen_vir = None

    #@property
    #def e_corr(self):
    #    """Best guess for correlation energy, using the lowest BNO threshold."""
    #    idx = np.argmin(self.bno_threshold)
    #    return self.e_corrs[idx]

    def init_orbital_plot(self):
        if self.boundary_cond == 'open':
            raise NotImplementedError()
        os.makedirs(self.opts.plot_orbitals_dir, exist_ok=True)
        name = "%s.cube" % os.path.join(self.opts.plot_orbitals_dir, self.id_name)
        nx, ny, nz = self.opts.plot_orbitals_gridsize
        cubefile = cubegen.CubeFile(self.mol, filename=name, nx=nx, ny=ny, nz=nz,
                **self.base.opts.plot_orbitals_kwargs)
        return cubefile

    def add_orbital_plot(self, name, mo_coeff=None, dm=None, dset_idx=None, keep_in_list=False):
        if mo_coeff is None and dm is None:
            raise ValueError("set mo_coeff or dm")
        if name in self.opts.plot_orbitals:
            if not keep_in_list:
                self.opts.plot_orbitals.remove(name)
            if mo_coeff is not None:
                self.log.debugv("Adding %s orbitals to cube file.", name)
                self.cubefile.add_orbital(mo_coeff.copy(), dset_idx=dset_idx)
            else:
                self.log.debugv("Adding %s density to cube file.", name)
                self.cubefile.add_density(dm.copy(), dset_idx=dset_idx)
            if not self.opts.plot_orbitals:
                self.write_orbital_plot()

    def write_orbital_plot(self):
        self.log.debug("Writing cube file.")
        self.cubefile.write()
        if self.opts.plot_orbitals_exit:
            raise self.Exit("All plots done")

    def make_bath(self, bath_type=NotSet):
        """Make DMET and MP2 bath natural orbitals."""
        if bath_type is NotSet:
            bath_type = self.opts.bath_type
        # Add fragment orbitals for cube file plots
        if self.opts.plot_orbitals:
            self.cubefile = self.init_orbital_plot()
            self.add_orbital_plot('fragment', self.c_frag)

        t0_bath = t0 = timer()
        self.log.info("Making DMET Bath")
        self.log.info("----------------")
        self.log.changeIndentLevel(1)
        c_dmet, c_env_occ, c_env_vir = self.make_dmet_bath(self.c_env, tol=self.opts.dmet_threshold)
        self.log.timing("Time for DMET bath:  %s", time_string(timer()-t0))
        # Add DMET orbitals for cube file plots
        if self.opts.plot_orbitals:
            self.add_orbital_plot('dmet', c_dmet, dset_idx=1001)
        self.log.changeIndentLevel(-1)

        # Add additional orbitals to cluster [optional]
        #c_dmet, c_env_occ, c_env_vir = self.additional_bath_for_cluster(c_dmet, c_env_occ, c_env_vir)

        # Diagonalize cluster DM to separate cluster occupied and virtual
        c_cluster_occ, c_cluster_vir = self.diagonalize_cluster_dm(self.c_frag, c_dmet, tol=2*self.opts.dmet_threshold)
        self.log.info("Cluster orbitals:  n(occ)= %3d  n(vir)= %3d", c_cluster_occ.shape[-1], c_cluster_vir.shape[-1])

        # Add cluster orbitals to plot
        if self.opts.plot_orbitals:
            self.add_orbital_plot('cluster', c_cluster_occ, dset_idx=2001, keep_in_list=True)
            self.add_orbital_plot('cluster', c_cluster_vir, dset_idx=3001)

        # Primary MP2 bath orbitals
        # TODO NOT MAINTAINED
        #if True:
        #    if self.opts.prim_mp2_bath_tol_occ:
        #        self.log.info("Adding primary occupied MP2 bath orbitals")
        #        C_add_o, C_rest_o, *_ = self.make_mp2_bath(C_occclst, C_virclst, "occ",
        #                c_occenv=C_occenv, c_virenv=C_virenv, tol=self.opts.prim_mp2_bath_tol_occ,
        #                mp2_correction=False)
        #    if self.opts.prim_mp2_bath_tol_vir:
        #        self.log.info("Adding primary virtual MP2 bath orbitals")
        #        C_add_v, C_rest_v, *_ = self.make_mp2_bath(C_occclst, C_virclst, "vir",
        #                c_occenv=C_occenv, c_virenv=C_virenv, tol=self.opts.prim_mp2_bath_tol_occ,
        #                mp2_correction=False)
        #    # Combine
        #    if self.opts.prim_mp2_bath_tol_occ:
        #        C_bath = np.hstack((C_add_o, C_bath))
        #        C_occenv = C_rest_o
        #    if self.opts.prim_mp2_bath_tol_vir:
        #        C_bath = np.hstack((C_bath, C_add_v))
        #        C_virenv = C_rest_v

        #    # Re-diagonalize cluster DM to separate cluster occupied and virtual
        #    C_occclst, C_virclst = self.diagonalize_cluster_dm(C_bath)
        #self.C_bath = C_bath

        self.log.debugv("bath_type= %r", bath_type)
        if bath_type is None or bath_type.upper() == 'NONE':
            c_no_occ = c_env_occ
            c_no_vir = c_env_vir
            n_no_occ = np.full((c_no_occ.shape[-1],), -np.inf)
            n_no_vir = np.full((c_no_vir.shape[-1],), -np.inf)
        elif bath_type.upper() == 'ALL':
            c_no_occ = c_env_occ
            c_no_vir = c_env_vir
            n_no_occ = np.full((c_no_occ.shape[-1],), np.inf)
            n_no_vir = np.full((c_no_vir.shape[-1],), np.inf)
        elif bath_type.upper() == 'MP2-BNO':
            c_no_occ, n_no_occ = self.make_bno_bath(c_cluster_occ, c_cluster_vir, c_env_occ, c_env_vir, 'occ')
            c_no_vir, n_no_vir = self.make_bno_bath(c_cluster_occ, c_cluster_vir, c_env_occ, c_env_vir, 'vir')
        else:
            raise ValueError("Unknown bath type: '%s'" % bath_type)


        #if self.opts.plot_orbitals:
        #    for key in self.opts.plot_orbitals.copy():
        #        if key.startswith('active-'):
        #            eta = float(key[key.find('[')+1:key.find(']')])
        #            act = (n_no >= eta)
        #            dm = np.dot(self.c_no_cc[:,mask], c_no[:,mask].T)

        self.log.timing("Time for bath:  %s", time_string(timer()-t0_bath))

        return c_cluster_occ, c_cluster_vir, c_no_occ, n_no_occ, c_no_vir, n_no_vir


    def make_bno_bath(self, c_cluster_occ, c_cluster_vir, c_env_occ, c_env_vir, kind):
        assert kind in ('occ', 'vir')
        c_env = c_env_occ if (kind == 'occ') else c_env_vir
        if c_env.shape[-1] == 0:
            return c_env, np.zeros((0,))

        name = {'occ': "occupied", 'vir': "virtual"}[kind]

        self.log.info("Making %s Bath NOs", name.capitalize())
        self.log.info("-------%s---------", len(name)*'-')
        self.log.changeIndentLevel(1)
        t0 = timer()
        c_no, n_no = make_mp2_bno(
                self, kind, c_cluster_occ, c_cluster_vir, c_env_occ, c_env_vir)
        self.log.debugv('BNO eigenvalues:\n%r', n_no)
        if len(n_no) > 0:
            self.log.info("%s Bath NO Histogram", name.capitalize())
            self.log.info("%s------------------", len(name)*'-')
            for line in helper.plot_histogram(n_no):
                self.log.info(line)
        # Orbital plot
        if self.opts.plot_orbitals:
            idx = 0
            for key in self.opts.plot_orbitals.copy():
                if key.startswith('bno-%s-' % kind):
                    itvl = key[key.find('[')+1:key.find(']')]
                    low, high = [float(x) for x in itvl.split(',')]
                    mask = np.logical_and(n_no >= low, n_no < high)
                    dm = np.dot(c_no[:,mask], c_no[:,mask].T)
                    self.add_orbital_plot(key, dm=dm, dset_idx=(4001 if kind=='occ' else 5001)+idx)
                    idx += 1
        self.log.timing("Time for %s BNOs:  %s", name, time_string(timer()-t0))
        self.log.changeIndentLevel(-1)

        return c_no, n_no


    def set_cas(self, iaos=None, c_occ=None, c_vir=None):
        """For TCCSD"""
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


    def kernel(self, bno_threshold=None, bno_number=None, solver=None, init_guess=None, eris=None):
        """Run solver for a single BNO threshold.

        Parameters
        ----------
        bno_threshold : float, optional
            Bath natural orbital (BNO) thresholds.
        bno_number : int, optional
            Number of bath natural orbitals. Default: None.
        solver : {'MP2', 'CISD', 'CCSD', 'CCSD(T)', 'FCI'}, optional
            Correlated solver.

        Returns
        -------
        results : self.Results
        """
        if (bno_threshold is None and bno_number is None):
            bno_threshold = self.base.bno_threshold

        if np.ndim(bno_threshold) == 0:
            bno_threshold = 2*[bno_threshold]
        if np.ndim(bno_number) == 0:
            bno_number = 2*[bno_number]

        if solver is None:
            solver = self.solver

        if self.c_cluster_occ is None:
            self.c_cluster_occ, self.c_cluster_vir, self.c_no_occ, self.n_no_occ, self.c_no_vir, self.n_no_vir = self.make_bath()

        #self.e_delta_mp2 = e_delta_occ + e_delta_vir
        #self.log.debug("MP2 correction = %.8g", self.e_delta_mp2)

        assert (self.c_no_occ is not None)
        assert (self.c_no_vir is not None)

        self.log.info("Occupied BNOs:")
        c_nbo_occ, c_frozen_occ = self.truncate_bno(self.c_no_occ, self.n_no_occ, bno_threshold[0], bno_number[0])
        self.log.info("Virtual BNOs:")
        c_nbo_vir, c_frozen_vir = self.truncate_bno(self.c_no_vir, self.n_no_vir, bno_threshold[1], bno_number[1])


        # Canonicalize orbitals
        c_active_occ = self.canonicalize_mo(self.c_cluster_occ, c_nbo_occ)[0]
        c_active_vir = self.canonicalize_mo(self.c_cluster_vir, c_nbo_vir)[0]
        # Do not overwrite self.c_active_occ/vir yet - we still need the previous coefficients
        # to generate an intial guess

        # TODO: Do not store these!
        self._c_frozen_occ = c_frozen_occ
        self._c_frozen_vir = c_frozen_vir

        # Active/frozen density plotting
        if 'active' in self.opts.plot_orbitals:
            dm = 2*(np.dot(c_active_occ, c_active_occ.T)
                  + np.dot(c_active_vir, c_active_vir.T))
            self.add_orbital_plot('active', dm=dm, dset_idx=(6001))
        if 'frozen' in self.opts.plot_orbitals:
            dm = 2*(np.dot(c_frozen_occ, c_frozen_occ.T)
                  + np.dot(c_frozen_vir, c_frozen_vir.T))
            self.add_orbital_plot('frozen', dm=dm, dset_idx=(7001))
        if self.opts.plot_orbitals:
            self.log.warning("The following orbital/densities could not be plotted: %r", self.opts.plot_orbitals)
            self.write_orbital_plot(cubefile)

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
        self.log.info("-------------" + len(str(self))*"-")
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
        self._c_active_occ = c_active_occ
        self._c_active_vir = c_active_vir

        if solver is None:
            return None

        # Create solver object
        t0 = timer()
        solver_opts = {}
        solver_opts['make_rdm1'] = self.opts.make_rdm1
        solver_opts['make_rdm2'] = self.opts.make_rdm2
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
            solver_opts['tcc_fci_opts'] = self.opts.tcc_fci_opts


        cluster_solver_cls = get_solver_class(solver)
        cluster_solver = cluster_solver_cls(self, mo_coeff, mo_occ, nocc_frozen=nocc_frozen, nvir_frozen=nvir_frozen, **solver_opts)
        if self.opts.nelectron_target is None:
            solver_results = cluster_solver.kernel(init_guess=init_guess, eris=eris)
        else:
            solver_results = cluster_solver.kernel_optimize_cpt(self.opts.nelectron_target, init_guess=init_guess, eris=eris)
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
        if bno_threshold[0] is not None:
            if bno_threshold[0] == bno_threshold[1]:
                self.log.info("BNO threshold= %.1e :  E(corr)= %+14.8f Ha", bno_threshold[0], e_corr)
            else:
                self.log.info("BNO threshold= %.1e / %.1e :  E(corr)= %+14.8f Ha", *bno_threshold, e_corr)
        else:
            self.log.info("BNO number= %3d / %3d:  E(corr)= %+14.8f Ha", *bno_number, e_corr)

        # --- Population analysis
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
                # Add frozen states and transform to AO
                dm1 = np.zeros(2*[self.base.nao])
                nocc = np.count_nonzero(self.mf.mo_occ > 0)
                dm1[np.diag_indices(nocc)] = 2
                a = cluster_solver.get_active_slice()
                dm1[a,a] = solver_results.dm1
                self.base.pop_analysis(dm1, mo_coeff=mo_coeff, filename=filename)
            except Exception as e:
                self.log.error("Exception in population analysis: %s", e)

        results = self.Results(
                fid=self.id,
                bno_threshold=bno_threshold,
                n_active=nactive,
                converged=solver_results.converged,
                e_corr=e_corr,
                dm1=solver_results.dm1, dm2=solver_results.dm2)

        #(results.t1_pf, results.t2_pf), (results.e1b, results.e2b_conn, results.e2b_disc) = self.project_solver_results(solver_results)

        # EOM analysis
        #if 'IP' in self.opts.eom_ccsd:
        #    results.ip_energy, _ = self.eom_analysis(cluster_solver, 'IP')
        #if 'EA' in self.opts.eom_ccsd:
        #    results.ea_energy, _ = self.eom_analysis(cluster_solver, 'EA')

        # Keep Amplitudes [optional]
        if self.base.opts.project_init_guess or self.opts.sc_mode:
            if hasattr(solver_results, 't2'):
                results.t1 = solver_results.t1
                results.t2 = solver_results.t2
            if hasattr(solver_results, 'c2'):
                results.c0 = solver_results.c0
                results.c1 = solver_results.c1
                results.c2 = solver_results.c2
        # Keep Lambda-Amplitudes
        if hasattr(solver_results, 'l2') and solver_results.l2 is not None:
            results.l1 = solver_results.l1
            results.l2 = solver_results.l2
        # Keep ERIs [optional]
        if self.base.opts.project_eris or self.opts.sc_mode:
            results.eris = solver_results.eris

        self._results = results

        # Force GC to free memory
        m0 = get_used_memory()
        del cluster_solver, solver_results
        ndel = gc.collect()
        self.log.debugv("GC deleted %d objects and freed %.3f MB of memory", ndel, (get_used_memory()-m0)/1e6)

        return results


    def truncate_bno(self, c_no, n_no, bno_threshold=None, bno_number=None):
        """Split natural orbitals (NO) into bath and rest."""
        if bno_number is not None:
            pass
        elif bno_threshold is not None:
            bno_threshold *= self.opts.bno_threshold_factor
            bno_number = np.count_nonzero(n_no >= bno_threshold)
        else:
            raise ValueError()

        # Logging
        fmt = "  > %4s: N= %4d  max= % 9.3g  min= % 9.3g  sum= % 9.3g ( %7.3f %%)"
        def log(name, n_part):
            if len(n_part) > 0:
                with np.errstate(invalid='ignore'): # supress 0/0=nan warning
                    self.log.info(fmt, name, len(n_part), max(n_part), min(n_part), np.sum(n_part),
                            100*np.sum(n_part)/np.sum(n_no))
            else:
                self.log.info(fmt[:fmt.index('max')].rstrip(), name, 0)
        log("Bath", n_no[:bno_number])
        log("Rest", n_no[bno_number:])

        c_bno, c_rest = np.hsplit(c_no, [bno_number])
        return c_bno, c_rest


    # Register frunctions of dmet_bath.py as methods
    #make_dmet_bath = make_dmet_bath


    def additional_bath_for_cluster(self, c_bath, c_occenv, c_virenv):
        """Add additional bath orbitals to cluster (fragment+DMET bath)."""
        # NOT MAINTAINED
        raise NotImplementedError()
        if self.power1_occ_bath_tol is not False:
            c_add, c_occenv, _ = make_mf_bath(self, c_occenv, "occ", bathtype="power",
                    tol=self.power1_occ_bath_tol)
            self.log.info("Adding %d first-order occupied power bath orbitals to cluster.", c_add.shape[-1])
            c_bath = np.hstack((c_add, c_bath))
        if self.power1_vir_bath_tol is not False:
            c_add, c_virenv, _ = make_mf_bath(self, c_virenv, "vir", bathtype="power",
                    tol=self.power1_vir_bath_tol)
            self.log.info("Adding %d first-order virtual power bath orbitals to cluster.", c_add.shape[-1])
            c_bath = np.hstack((c_bath, c_add))
        # Local orbitals:
        if self.local_occ_bath_tol is not False:
            c_add, c_occenv = make_local_bath(self, c_occenv, tol=self.local_occ_bath_tol)
            self.log.info("Adding %d local occupied bath orbitals to cluster.", c_add.shape[-1])
            c_bath = np.hstack((c_add, c_bath))
        if self.local_vir_bath_tol is not False:
            c_add, c_virenv = make_local_bath(self, c_virenv, tol=self.local_vir_bath_tol)
            self.log.info("Adding %d local virtual bath orbitals to cluster.", c_add.shape[-1])
            c_bath = np.hstack((c_bath, c_add))
        return c_bath, c_occenv, c_virenv

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

    def project_solver_results(self, results):
        # Projected amplitudes
        rf = dot(self.c_frag.T, self.base.get_ovlp(), self.c_active_occ)
        t1, t2 = results.t1, results.t2
        t1_pf = np.dot(rf, t1)
        t2_pf = np.tensordot(rf, t2, axes=1)
        #t2_pf = (t2_pf + t2_pf.transpose(1,0,3,2)) / 2
        # --- Correlation energy
        eris = results.eris
        nocc, nvir = t2_pf.shape[1:3]
        # E(1-body)
        fov = np.dot(rf, eris.fock[:nocc,nocc:])
        e1b = 2*np.sum(fov * t1_pf)
        # E(2-body)
        tau = t2_pf
        if hasattr(eris, 'ovvo'):
            gov = eris.ovvo[:]
        elif hasattr(eris, 'ovov'):
            # MP2 only has eris.ovov - for real integrals we transpose
            gov = eris.ovov[:].reshape(nocc,nvir,nocc,nvir).transpose(0, 1, 3, 2).conj()
        #else:
        #    g_ovvo = eris[occ,vir,vir,occ]
        gov1 = np.tensordot(rf, gov, axes=1)
        gov2 = einsum('xj,iabj->xabi', rf, gov)
        #e2 = 2*einsum('ijab,iabj', t2_pf, gov) - einsum('ijab,jabi', t2_pf, gov)
        #e2 = 2*einsum('ijab,iabj', t2_pf, gov) - einsum('ijab,ibaj', t2_pf, gov)
        #gov = (2*gov + gov.transpose(0, 2, 1, 3))
        gov = (2*gov1 - gov2)

        #e2 = 2*einsum('ijab,iabj', p2, g_ovvo) - einsum('ijab,jabi', p2, g_ovvo)
        e2b_conn = einsum('ijab,iabj->', t2_pf, gov)
        #e2_t1 = einsum('ia,jb,iabj->', t1_pf, t1, gov)
        e2b_disc = einsum('ia,iabj->jb', t1_pf, gov)
        #e2b_disc = 0.0
        #self.log.info("Energy components: E[C1]= % 16.8f Ha, E[C2]= % 16.8f Ha", e1, e2)
        #if e1 > 1e-4 and 10*e1 > e2:
        #    self.log.warning("WARNING: Large E[C1] component!")
        #e_frag = self.opts.energy_factor * self.sym_factor * (e1 + e2)
        return (t1_pf, t2_pf), (e1b, e2b_conn, e2b_disc)

    def get_fragment_energy(self, p1, p2, eris):
        """Calculate fragment correlation energy contribution from porjected C1, C2.

        Parameters
        ----------
        p1 : (n(occ), n(vir)) array
            Locally projected C1 amplitudes.
        p2 : (n(occ), n(occ), n(vir), n(vir)) array
            Locally projected C2 amplitudes.
        eris :
            PySCF eris object as returned by cm.ao2mo()

        Returns
        -------
        e_frag : float
            Fragment energy contribution.
        """
        if self.opts.energy_factor == 0:
            return 0

        nocc, nvir = p2.shape[1:3]
        occ = np.s_[:nocc]
        vir = np.s_[nocc:]
        # E1
        e1 = 0
        if p1 is not None:
            if hasattr(eris, 'fock'):
                f = eris.fock[occ,vir]
            else:
                f = np.linalg.multi_dot((self.c_active_occ.T, self.base.get_fock(), self.c_active_vir))
            e1 = 2*np.sum(f * p1)
        # E2
        if hasattr(eris, 'ovvo'):
            g_ovvo = eris.ovvo[:]
        elif hasattr(eris, 'ovov'):
            # MP2 only has eris.ovov - for real integrals we transpose
            g_ovvo = eris.ovov[:].reshape(nocc,nvir,nocc,nvir).transpose(0, 1, 3, 2).conj()
        else:
            g_ovvo = eris[occ,vir,vir,occ]

        e2 = 2*einsum('ijab,iabj', p2, g_ovvo) - einsum('ijab,jabi', p2, g_ovvo)
        self.log.info("Energy components: E[C1]= % 16.8f Ha, E[C2]= % 16.8f Ha", e1, e2)
        if e1 > 1e-4 and 10*e1 > e2:
            self.log.warning("WARNING: Large E[C1] component!")
        e_frag = self.opts.energy_factor * self.sym_factor * (e1 + e2)
        return e_frag

    #def get_fragment_dmet_energy(self, p_frag, dm1=None, dm2=None, eris=None):
    #    if dm1 is None:
    #        dm1 = self.results.dm1
    #    if dm2 is None:
    #        dm2 = self.results.dm2
    #    if eris is None:
    #        eris = self.results.eris
    #    if None in [dm1, dm2, eris]:
    #        self.log.error("Calculation of DMET energy requires dm1, dm2, and eris")
    #        return None
    #    h1e = self.base.get_hcore()
    #    e1 = einsum('ij,ix,xj->', h1e, p_frag, dm1)
    #    e2 = einsum('ijkl,ix,xj->', eris, p_frag, dm1)


    def eom_analysis(self, csolver, kind, filename=None, mode="a", sort_weight=True, r1_min=1e-2):
        kind = kind.upper()
        assert kind in ("IP", "EA")

        if filename is None:
            filename = "%s-%s.txt" % (self.base.opts.eomfile, self.name)

        sc = np.dot(self.base.get_ovlp(), self.base.lo)
        if kind == "IP":
            e, c = csolver.ip_energy, csolver.ip_coeff
        elif kind == "EA":
            e, c = csolver.ea_energy, csolver.ea_coeff
        else:
            raise ValueError()
        nroots = len(e)
        eris = csolver._eris
        cc = csolver._solver

        self.log.info("EOM-CCSD %s energies= %r", kind, e[:5].tolist())
        tstamp = datetime.now()
        self.log.info("[%s] Writing detailed cluster %s-EOM analysis to file \"%s\"", tstamp, kind, filename)

        with open(filename, mode) as f:
            f.write("[%s] %s-EOM analysis\n" % (tstamp, kind))
            f.write("*%s*****************\n" % (26*"*"))

            for root in range(nroots):
                r1 = c[root][:cc.nocc]
                qp = np.linalg.norm(r1)**2
                f.write("  %s-EOM-CCSD root= %2d , energy= %+16.8g , QP-weight= %10.5g\n" %
                        (kind, root, e[root], qp))
                if qp < 0.0 or qp > 1.0:
                    self.log.error("Error: QP-weight not between 0 and 1!")
                r1lo = einsum("i,ai,al->l", r1, eris.mo_coeff[:,:cc.nocc], sc)

                if sort_weight:
                    order = np.argsort(-r1lo**2)
                    for ao, lab in enumerate(np.asarray(self.mf.mol.ao_labels())[order]):
                        wgt = r1lo[order][ao]**2
                        if wgt < r1_min*qp:
                            break
                        f.write("  * Weight of %s root %2d on OrthAO %-16s = %10.5f\n" %
                                (kind, root, lab, wgt))
                else:
                    for ao, lab in enumerate(ao_labels):
                        wgt = r1lo[ao]**2
                        if wgt < r1_min*qp:
                            continue
                        f.write("  * Weight of %s root %2d on OrthAO %-16s = %10.5f\n" %
                                (kind, root, lab, wgt))

        return e, c


    def get_fragment_bsse(self, rmax=None, nimages=5, unit='A'):
        self.log.info("Counterpoise Calculation")
        self.log.info("************************")
        # Currently only PBC
        #if not self.boundary_cond == 'open':
        #    raise NotImplementedError()
        if rmax is None:
            rmax = self.opts.bsse_rmax

        # Atomic calculation with atomic basis functions:
        #mol = self.mol.copy()
        #atom = mol.atom[self.atoms]
        #self.log.debugv("Keeping atoms %r", atom)
        #mol.atom = atom
        #mol.a = None
        #mol.build(False, False)

        natom0, e_mf0, e_cm0, dm = self.counterpoise_calculation(rmax=0.0, nimages=0)
        assert natom0 == len(self.atoms)
        self.log.debugv("Counterpoise: E(atom)= % 16.8f Ha", e_cm0)

        #natom_list = []
        #e_mf_list = []
        #e_cm_list = []
        r_values = np.hstack((np.arange(1.0, int(rmax)+1, 1.0), rmax))
        #for r in r_values:
        r = rmax
        natom, e_mf, e_cm, dm = self.counterpoise_calculation(rmax=r, dm0=dm)
        self.log.debugv("Counterpoise: n(atom)= %3d  E(mf)= %16.8f Ha  E(%s)= % 16.8f Ha", natom, e_mf, self.solver, e_cm)

        e_bsse = self.sym_factor*(e_cm - e_cm0)
        self.log.debugv("Counterpoise: E(BSSE)= % 16.8f Ha", e_bsse)
        return e_bsse


    def counterpoise_calculation(self, rmax, dm0=None, nimages=5, unit='A'):
        mol = self.make_counterpoise_mol(rmax, nimages=nimages, unit=unit, output='pyscf-cp.txt')
        # Mean-field
        #mf = type(self.mf)(mol)
        mf = pyscf.scf.RHF(mol)
        mf.conv_tol = self.mf.conv_tol
        #if self.mf.with_df is not None:
        #    self.log.debugv("Setting GDF")
        #    self.log.debugv("%s", type(self.mf.with_df))
        #    # ONLY GDF SO FAR!
        # TODO: generalize
        if self.base.kdf is not None:
            auxbasis = self.base.kdf.auxbasis
        elif self.mf.with_df is not None:
            auxbasis = self.mf.with_df.auxbasis
        else:
            auxbasis=None
        if auxbasis:
            mf = mf.density_fit(auxbasis=auxbasis)
        # TODO:
        #use dm0 as starting point
        mf.kernel()
        dm0 = mf.make_rdm1()
        # Embedded calculation with same options
        ecc = ewf.EWF(mf, solver=self.solver, bno_threshold=self.bno_threshold, options=self.base.opts)
        ecc.make_atom_cluster(self.atoms, options=self.opts)
        ecc.kernel()

        return mol.natm, mf.e_tot, ecc.e_tot, dm0
