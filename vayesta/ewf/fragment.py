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
from vayesta.core.fragmentation import IAO_Fragmentation

from vayesta.core.qemb.bath import DMET_Bath, CompleteBath

from . import ewf
from .bath import BNO_Bath
from . import helper
from . import psubspace

class Cluster:

    def __init__(self):
        self._c_active_occ = None
        self._c_active_vir = None
        self._c_frozen_occ = None
        self._c_frozen_vir = None
        self.sym_op = None
        self.sym_parent = None

    @property
    def nmo(self):
        return self.nocc + self.nvir

    @property
    def nocc(self):
        return self.n_active_occ + self.n_frozen_occ

    @property
    def nvir(self):
        return self.n_active_vir + self.n_frozen_vir

    # --- Active

    @property
    def c_active(self):
        """Active orbital coefficients."""
        if self.c_active_occ is None:
            return None
        return hstack(self.c_active_occ, self.c_active_vir)

    @property
    def c_active_occ(self):
        """Active occupied orbital coefficients."""
        if self.sym_parent is None:
            return self._c_active_occ
        else:
            return self.sym_op(self.sym_parent.c_active_occ)

    @property
    def c_active_vir(self):
        """Active virtual orbital coefficients."""
        if self.sym_parent is None:
            return self._c_active_vir
        else:
            return self.sym_op(self.sym_parent.c_active_vir)

    @property
    def nactive(self):
        """Number of active orbitals."""
        return (self.n_active_occ + self.n_active_vir)

    @property
    def nactive_occ(self):
        """Number of active occupied orbitals."""
        return self.c_active_occ.shape[-1]

    @property
    def nactive_vir(self):
        """Number of active virtual orbitals."""
        return self.c_active_vir.shape[-1]

    # --- Frozen

    @property
    def c_frozen(self):
        """Frozen orbital coefficients."""
        if self.c_frozen_occ is None:
            return None
        return hstack(self.c_frozen_occ, self.c_frozen_vir)

    @property
    def c_frozen_occ(self):
        """Frozen occupied orbital coefficients."""
        if self.sym_parent is None:
            return self._c_frozen_occ
        else:
            return self.sym_op(self.sym_parent.c_frozen_occ)

    @property
    def c_frozen_vir(self):
        """Frozen virtual orbital coefficients."""
        if self.sym_parent is None:
            return self._c_frozen_vir
        else:
            return self.sym_op(self.sym_parent.c_frozen_vir)

    @property
    def nfrozen(self):
        """Number of frozen orbitals."""
        return (self.n_frozen_occ + self.n_frozen_vir)

    @property
    def nfrozen_occ(self):
        """Number of frozen occupied orbitals."""
        return self.c_frozen_occ.shape[-1]

    @property
    def nfrozen_vir(self):
        """Number of frozen virtual orbitals."""
        return self.c_frozen_vir.shape[-1]


class EWFFragment(QEmbeddingFragment):

    @dataclasses.dataclass
    class Options(QEmbeddingFragment.Options):
        """Attributes set to `NotSet` inherit their value from the parent EWF object."""
        # Options also present in `base`:
        dmet_threshold: float = NotSet
        make_rdm1: bool = NotSet
        make_rdm2: bool = NotSet
        solve_lambda: bool = NotSet                 # If False, use T-amplitudes inplace of Lambda-amplitudes
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
        bno_number: int = None         # Set a fixed number of BNOs
        # Additional fragment specific options:
        bno_threshold_factor: float = 1.0
        # CAS methods
        c_cas_occ: np.ndarray = None
        c_cas_vir: np.ndarray = None
        #
        calculate_e_dmet: bool = 'auto'
        #
        dm_with_frozen: bool = NotSet
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

        # For self-consistent mode
        self.solver_results = None
        # For orbital plotting
        self.cubefile = None

    #@property
    #def e_corr(self):
    #    """Best guess for correlation energy, using the lowest BNO threshold."""
    #    idx = np.argmin(self.bno_threshold)
    #    return self.e_corrs[idx]

    @property
    def c_cluster_occ(self):
        return self.bath.c_cluster_occ

    @property
    def c_cluster_vir(self):
        return self.bath.c_cluster_vir

    def reset(self):
        super().reset()

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

    def set_cas(self, iaos=None, c_occ=None, c_vir=None, minao='auto', dmet_threshold=None):
        """Set complete active space for tailored CCSD"""
        if dmet_threshold is None:
            dmet_threshold = 2*self.opts.dmet_threshold
        if iaos is not None:
            if isinstance(self.base.fragmentation, IAO_Fragmentation):
                fragmentation = self.base.fragmentation
            # Create new IAO fragmentation
            else:
                fragmentation = IAO_Fragmentation(self, minao=minao).kernel()
            # Get IAO and environment coefficients from fragmentation
            indices = fragmentation.get_orbital_fragment_indices(iaos)[1]
            c_iao = fragmentation.get_frag_coeff(indices)
            c_env = fragmentation.get_env_coeff(indices)
            bath = DMET_Bath(self, dmet_threshold=dmet_threshold)
            c_dmet = bath.make_dmet_bath(c_env)[0]
            c_iao_occ, c_iao_vir = self.diagonalize_cluster_dm(c_iao, c_dmet, tol=2*dmet_threshold)
        else:
            c_iao_occ = c_iao_vir = None

        c_cas_occ = hstack(c_occ, c_iao_occ)
        c_cas_vir = hstack(c_vir, c_iao_vir)
        self.opts.c_cas_occ = c_cas_occ
        self.opts.c_cas_vir = c_cas_vir
        return c_cas_occ, c_cas_vir

    def make_bath(self, bath_type=NotSet):
        if bath_type is NotSet:
            bath_type = self.opts.bath_type
        # DMET bath only
        if bath_type is None or bath_type.lower() == 'dmet':
            bath = DMET_Bath(self, dmet_threshold=self.opts.dmet_threshold)
        # All environment orbitals as bath
        elif bath_type.lower() in ('all', 'complete'):
            bath = CompleteBath(self, dmet_threshold=self.opts.dmet_threshold)
        # MP2 bath natural orbitals
        elif bath_type.lower() == 'mp2-bno':
            bath = BNO_Bath(self, dmet_threshold=self.opts.dmet_threshold)

        bath.kernel()

        self.bath = bath
        return bath

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
        if bno_number is None:
            bno_number = self.opts.bno_number
        if bno_number is None and bno_threshold is None:
            bno_threshold = self.base.bno_threshold
        if np.ndim(bno_threshold) == 0:
            bno_threshold = 2*[bno_threshold]
        if np.ndim(bno_number) == 0:
            bno_number = 2*[bno_number]
        if solver is None:
            solver = self.solver
        if self.bath is None:
            self.make_bath()

        c_bno_occ, c_frozen_occ = self.bath.get_occupied_bath(bno_threshold[0], bno_number[0])
        c_bno_vir, c_frozen_vir = self.bath.get_virtual_bath(bno_threshold[1], bno_number[1])

        # Canonicalize orbitals
        c_active_occ = self.canonicalize_mo(self.bath.c_cluster_occ, c_bno_occ)[0]
        c_active_vir = self.canonicalize_mo(self.bath.c_cluster_vir, c_bno_vir)[0]
        # Do not overwrite self.c_active_occ/vir yet - we still need the previous coefficients
        # to generate an intial guess

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
        c_occ = self.stack_mo(c_frozen_occ, c_active_occ)
        c_vir = self.stack_mo(c_active_vir, c_frozen_vir)
        mo_coeff = self.stack_mo(c_occ, c_vir)

        # Check occupations
        # TODO: Clean this
        self.check_mo_occupation((2 if self.base.is_rhf else 1), c_occ)
        self.check_mo_occupation(0, c_vir)
        if self.base.is_rhf:
            nocc, nvir = c_occ.shape[-1], c_vir.shape[-1]
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

        else:
            mo_occ = []
            for s, spin in enumerate(('alpha', 'beta')):
                nocc, nvir = c_occ[s].shape[-1], c_vir[s].shape[-1]
                mo_occ.append(np.asarray(nocc*[2] + nvir*[0]))
                self.log.info("Orbitals for %s", self)
                self.log.info("-------------" + len(str(self))*"-")
                self.log.info("  > Active:   n(occ)= %5d  n(vir)= %5d  n(tot)= %5d", c_active_occ.shape[-1], c_active_vir.shape[-1], nactive)
                self.log.info("  > Frozen:   n(occ)= %5d  n(vir)= %5d  n(tot)= %5d", nocc_frozen, nvir_frozen, nfrozen)
                self.log.info("  > Total:    n(occ)= %5d  n(vir)= %5d  n(tot)= %5d", c_occ.shape[-1], c_vir.shape[-1], mo_coeff.shape[-1])
            mo_occ = tuple(mo_occ)

        # SPLIT HERE?

        # --- Project initial guess and integrals from previous cluster calculation with smaller eta:
        # Use initial guess from previous calculations
        # For self-consistent calculations, we can restart calculation:
        if init_guess is None and 'ccsd' in solver.lower():
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
                p_occ = dot(self.c_active_occ.T, self.base.get_ovlp(), c_active_occ)
                p_vir = dot(self.c_active_vir.T, self.base.get_ovlp(), c_active_vir)
                #t1, t2 = init_guess.pop('t1'), init_guess.pop('t2')
                t1, t2 = helper.transform_amplitudes(self.results.t1, self.results.t2, p_occ, p_vir)
                init_guess = {'t1' : t1, 't2' : t2}
        if init_guess is None: init_guess = {}

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
        solver_cls = get_solver_class(self.mf, solver)
        solver_opts = self.get_solver_options(solver)
        cluster_solver = solver_cls(self, mo_coeff, mo_occ, nocc_frozen=nocc_frozen, nvir_frozen=nvir_frozen, **solver_opts)
        if self.opts.nelectron_target is not None:
            cluster_solver.optimize_cpt(self.opts.nelectron_target, c_frag=self.c_proj)
        if eris is None:
            eris = cluster_solver.get_eris()
        solver_results = cluster_solver.kernel(eris=eris, **init_guess)

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

        #e_corr = self.get_fragment_energy(p1, p2, eris=solver_results.eris)
        e_corr = self.get_fragment_energy(p1, p2, eris=eris)
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
            results.eris = eris

        self._results = results

        # DMET energy
        calc_dmet = self.opts.calculate_e_dmet
        if calc_dmet == 'auto':
            calc_dmet = (solver_results.dm1 is not None and solver_results.dm2 is not None)
        if calc_dmet:
            results.e_dmet = self.get_fragment_dmet_energy(dm1=results.dm1, dm2=results.dm2, eris=eris)

        # Force GC to free memory
        #m0 = get_used_memory()
        #del cluster_solver, solver_results
        #ndel = gc.collect()
        #self.log.debugv("GC deleted %d objects and freed %.3f MB of memory", ndel, (get_used_memory()-m0)/1e6)

        return results

    def get_solver_options(self, solver):
        # TODO: fix this mess...
        solver_opts = {}
        solver_opts.update(self.opts.solver_options)
        pass_through = ['make_rdm1', 'make_rdm2']
        if 'CCSD' in solver.upper():
            pass_through += ['solve_lambda', 'sc_mode', 'dm_with_frozen', 'eom_ccsd', 'eom_ccsd_nroots']
        for attr in pass_through:
            self.log.debugv("Passing fragment option %s to solver.", attr)
            solver_opts[attr] = getattr(self.opts, attr)

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
        return solver_opts

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

    #def project_solver_results(self, results):
    #    # Projected amplitudes
    #    rf = dot(self.c_frag.T, self.base.get_ovlp(), self.c_active_occ)
    #    t1, t2 = results.t1, results.t2
    #    t1_pf = np.dot(rf, t1)
    #    t2_pf = np.tensordot(rf, t2, axes=1)
    #    #t2_pf = (t2_pf + t2_pf.transpose(1,0,3,2)) / 2
    #    # --- Correlation energy
    #    eris = results.eris
    #    nocc, nvir = t2_pf.shape[1:3]
    #    # E(1-body)
    #    fov = np.dot(rf, eris.fock[:nocc,nocc:])
    #    e1b = 2*np.sum(fov * t1_pf)
    #    # E(2-body)
    #    tau = t2_pf
    #    if hasattr(eris, 'ovvo'):
    #        gov = eris.ovvo[:]
    #    elif hasattr(eris, 'ovov'):
    #        # MP2 only has eris.ovov - for real integrals we transpose
    #        gov = eris.ovov[:].reshape(nocc,nvir,nocc,nvir).transpose(0, 1, 3, 2).conj()
    #    #else:
    #    #    g_ovvo = eris[occ,vir,vir,occ]
    #    gov1 = np.tensordot(rf, gov, axes=1)
    #    gov2 = einsum('xj,iabj->xabi', rf, gov)
    #    #e2 = 2*einsum('ijab,iabj', t2_pf, gov) - einsum('ijab,jabi', t2_pf, gov)
    #    #e2 = 2*einsum('ijab,iabj', t2_pf, gov) - einsum('ijab,ibaj', t2_pf, gov)
    #    #gov = (2*gov + gov.transpose(0, 2, 1, 3))
    #    gov = (2*gov1 - gov2)

    #    #e2 = 2*einsum('ijab,iabj', p2, g_ovvo) - einsum('ijab,jabi', p2, g_ovvo)
    #    e2b_conn = einsum('ijab,iabj->', t2_pf, gov)
    #    #e2_t1 = einsum('ia,jb,iabj->', t1_pf, t1, gov)
    #    e2b_disc = einsum('ia,iabj->jb', t1_pf, gov)
    #    #e2b_disc = 0.0
    #    #self.log.info("Energy components: E[C1]= % 16.8f Ha, E[C2]= % 16.8f Ha", e1, e2)
    #    #if e1 > 1e-4 and 10*e1 > e2:
    #    #    self.log.warning("WARNING: Large E[C1] component!")
    #    #e_frag = self.opts.energy_factor * self.sym_factor * (e1 + e2)
    #    return (t1_pf, t2_pf), (e1b, e2b_conn, e2b_disc)

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
                f = dot(self.c_active_occ.T, self.base.get_fock(), self.c_active_vir)
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
