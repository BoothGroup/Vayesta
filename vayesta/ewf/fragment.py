# Standard libaries
from datetime import datetime
import dataclasses
from typing import Union

# External libaries
import numpy as np

# Internal libaries
import pyscf
import pyscf.pbc
import pyscf.cc

# Local modules
import vayesta
from vayesta.core.util import *
from vayesta.core.qemb import Fragment as BaseFragment
from vayesta.solver import get_solver_class
from vayesta.core.fragmentation import IAO_Fragmentation
from vayesta.core.types import RFCI_WaveFunction

from vayesta.core.bath import BNO_Threshold
from vayesta.core.bath import DMET_Bath
from vayesta.core.bath import EwDMET_Bath
from vayesta.core.bath import BNO_Bath
from vayesta.core.bath import MP2_BNO_Bath
from vayesta.core.bath import CompleteBath
from vayesta.core.types import Cluster
from vayesta.core import ao2mo
from vayesta.core.mpi import mpi

from . import ewf

# Get MPI rank of fragment
get_fragment_mpi_rank = lambda *args : args[0].mpi_rank

class Fragment(BaseFragment):

    @dataclasses.dataclass
    class Options(BaseFragment.Options):
        """Attributes set to `NotSet` inherit their value from the parent EWF object."""
        # Options also present in `base`:
        dmet_threshold: float = NotSet
        solve_lambda: bool = NotSet
        t_as_lambda: bool = NotSet                  # If True, use T-amplitudes inplace of Lambda-amplitudes
        bsse_correction: bool = NotSet
        bsse_rmax: float = NotSet
        energy_factor: float = 1.0
        #energy_partitioning: str = NotSet
        sc_mode: int = NotSet
        nelectron_target: int = NotSet                  # If set, adjust bath chemical potential until electron number in fragment equals nelectron_target
        # Bath
        bath_type: str = NotSet
        bno_truncation: str = NotSet                    # Type of BNO truncation ["occupation", "number", "excited-percent", "electron-percent"]
        bno_threshold: float = NotSet
        bno_threshold_occ: float = NotSet
        bno_threshold_vir: float = NotSet
        bno_project_t2: bool = NotSet
        ewdmet_max_order: int = NotSet
        # CAS methods
        c_cas_occ: np.ndarray = None
        c_cas_vir: np.ndarray = None
        # --- Solver options
        tcc_fci_opts: dict = dataclasses.field(default_factory=dict)
        # --- Intercluster MP2 energy
        icmp2_bno_threshold: float = NotSet
        # --- Couple embedding problems (currently only CCSD and MPI)
        coupled_iterations: bool = NotSet

    @dataclasses.dataclass
    class Results(BaseFragment.Results):
        n_active: int = None
        ip_energy: np.ndarray = None
        ea_energy: np.ndarray = None

        @property
        def dm1(self):
            """Cluster 1DM"""
            return self.wf.make_rdm1()

        @property
        def dm2(self):
            """Cluster 2DM"""
            return self.wf.make_rdm2()

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

        # Default options:
        #defaults = self.Options().replace(self.base.Options(), select=NotSet)
        #for key, val in self.opts.items():
        #    if val != getattr(defaults, key):
        #        self.log.info('  > %-24s %3s %r', key + ':', '(*)', val)
        #    else:
        #        self.log.debugv('  > %-24s %3s %r', key + ':', '', val)

        if solver is None:
            solver = self.base.solver
        if solver not in self.base.valid_solvers:
            raise ValueError("Unknown solver: %s" % solver)
        self.solver = solver

        # For self-consistent mode
        self.solver_results = None

    @property
    def c_cluster_occ(self):
        return self.bath.c_cluster_occ

    @property
    def c_cluster_vir(self):
        return self.bath.c_cluster_vir

    def set_cas(self, iaos=None, c_occ=None, c_vir=None, minao='auto', dmet_threshold=None):
        """Set complete active space for tailored CCSD"""
        if dmet_threshold is None:
            dmet_threshold = 2*self.opts.dmet_threshold
        if iaos is not None:
            if isinstance(self.base.fragmentation, IAO_Fragmentation):
                fragmentation = self.base.fragmentation
            # Create new IAO fragmentation
            else:
                fragmentation = IAO_Fragmentation(self, minao=minao)
                fragmentation.kernel()
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
        """TODO: move to embedding base class?"""
        if bath_type is NotSet:
            bath_type = self.opts.bath_type
        if bath_type is None:
            self.log.warning("bath_type = None is deprecated; use bath_type = 'dmet'.")
            bath_type = 'dmet'
        if bath_type.lower() == 'all':
            self.log.warning("bath_type = 'all' is deprecated; use bath_type = 'full'.")
            bath_type = 'full'

        # All environment orbitals as bath (for testing purposes)
        if bath_type.lower() == 'full':
            self.bath = CompleteBath(self, dmet_threshold=self.opts.dmet_threshold)
            self.bath.kernel()
            return self.bath
        dmet_bath = DMET_Bath(self, dmet_threshold=self.opts.dmet_threshold)
        dmet_bath.kernel()
        # DMET bath only
        if bath_type.lower() == 'dmet':
            self.bath = dmet_bath
            return self.bath
        # Energy-weighted (Ew) DMET bath
        if bath_type.lower() == 'ewdmet':
            self.bath = EwDMET_Bath(self, dmet_bath, max_order=self.opts.ewdmet_max_order)
            self.bath.kernel()
            return self.bath
        # MP2 bath natural orbitals
        if bath_type.lower() == 'mp2-bno':
            project_t2 = self.opts.bno_project_t2 if hasattr(self.opts, 'bno_project_t2') else False
            self.bath = MP2_BNO_Bath(self, ref_bath=dmet_bath, project_t2=project_t2)
            self.bath.kernel()
            return self.bath
        if bath_type.lower() == 'mp2-bno-ewdmet':
            ewdmet_bath = EwDMET_Bath(self, dmet_bath, max_order=self.opts.ewdmet_max_order)
            ewdmet_bath.kernel()
            project_t2 = self.opts.bno_project_t2 if hasattr(self.opts, 'bno_project_t2') else False
            self.bath = MP2_BNO_Bath(self, ref_bath=ewdmet_bath, project_t2=project_t2)
            self.bath.kernel()
            return self.bath
        raise ValueError("Unknown bath_type: %r" % bath_type)

    def make_cluster(self, bath=None, bno_threshold=None, bno_threshold_occ=None, bno_threshold_vir=None):
        if bath is None:
            bath = self.bath
        if bath is None:
            raise ValueError("make_cluster requires bath.")
        if bno_threshold_occ is None:
            bno_threshold_occ = bno_threshold
        if bno_threshold_vir is None:
            bno_threshold_vir = bno_threshold
        c_bath_occ, c_frozen_occ = bath.get_occupied_bath(bno_threshold=bno_threshold_occ)
        c_bath_vir, c_frozen_vir = bath.get_virtual_bath(bno_threshold=bno_threshold_vir)
        # Canonicalize orbitals
        c_active_occ = self.canonicalize_mo(bath.dmet_bath.c_cluster_occ, c_bath_occ)[0]
        c_active_vir = self.canonicalize_mo(bath.dmet_bath.c_cluster_vir, c_bath_vir)[0]
        cluster = Cluster.from_coeffs(c_active_occ, c_active_vir, c_frozen_occ, c_frozen_vir)

        def check_occupation(mo_coeff, expected):
            occup = self.get_mo_occupation(mo_coeff)
            # RHF
            if np.ndim(occup[0]) == 0:
                assert np.allclose(occup, 2*expected, rtol=0, atol=2*self.opts.dmet_threshold)
            else:
                assert np.allclose(occup[0], expected, rtol=0, atol=self.opts.dmet_threshold)
                assert np.allclose(occup[1], expected, rtol=0, atol=self.opts.dmet_threshold)

        check_occupation(cluster.c_total_occ, 1)
        check_occupation(cluster.c_total_vir, 0)

        #self.cluster = cluster
        return cluster

    def make_bath_and_cluster(self, bno_threshold=None, bno_threshold_occ=None, bno_threshold_vir=None):
        """Run solver for a single BNO threshold.

        Parameters
        ----------
        bno_threshold : float, optional
            Bath natural orbital (BNO) threshold.

        Returns
        -------
        cluster
        """
        if bno_threshold is None:
            bno_threshold = self.opts.bno_threshold
        if bno_threshold_occ is None:
            bno_threshold_occ = self.opts.bno_threshold_occ
        if bno_threshold_vir is None:
            bno_threshold_vir = self.opts.bno_threshold_vir

        bno_threshold = BNO_Threshold(self.opts.bno_truncation, bno_threshold)

        if bno_threshold_occ is not None:
            bno_threshold_occ = BNO_Threshold(self.opts.bno_truncation, bno_threshold_occ)
        if bno_threshold_vir is not None:
            bno_threshold_vir = BNO_Threshold(self.opts.bno_truncation, bno_threshold_vir)

        if self.bath is None:
            self.make_bath()

        cluster = self.make_cluster(self.bath, bno_threshold=bno_threshold,
                bno_threshold_occ=bno_threshold_occ, bno_threshold_vir=bno_threshold_vir)
        self.log.info('Orbitals for %s', self)
        self.log.info('-------------%s', len(str(self))*'-')
        self.log.info(cluster.repr_size().replace('%', '%%'))
        self.cluster = cluster
        return cluster

    def get_init_guess(self, init_guess, solver, cluster):
        # FIXME
        return {}
        # --- Project initial guess and integrals from previous cluster calculation with smaller eta:
        # Use initial guess from previous calculations
        # For self-consistent calculations, we can restart calculation:
        #if init_guess is None and 'ccsd' in solver.lower():
        #    if self.base.opts.sc_mode and self.base.iteration > 1:
        #        self.log.debugv("Restarting using T1,T2 from previous iteration")
        #        init_guess = {'t1' : self.results.t1, 't2' : self.results.t2}
        #    elif self.base.opts.project_init_guess and self.results.t2 is not None:
        #        self.log.debugv("Restarting using projected previous T1,T2")
        #        # Projectors for occupied and virtual orbitals
        #        p_occ = dot(self.c_active_occ.T, self.base.get_ovlp(), cluster.c_active_occ)
        #        p_vir = dot(self.c_active_vir.T, self.base.get_ovlp(), cluster.c_active_vir)
        #        #t1, t2 = init_guess.pop('t1'), init_guess.pop('t2')
        #        t1, t2 = helper.transform_amplitudes(self.results.t1, self.results.t2, p_occ, p_vir)
        #        init_guess = {'t1' : t1, 't2' : t2}
        #if init_guess is None: init_guess = {}
        #return init_guess

    #def kernel(self, bno_threshold=None, bno_threshold_occ=None, bno_threshold_vir=None, solver=None, init_guess=None, eris=None):
        #"""Run solver for a single BNO threshold.

        #Parameters
        #----------
        #bno_threshold : float, optional
        #    Bath natural orbital (BNO) threshold.
        #solver : {'MP2', 'CISD', 'CCSD', 'FCI'}, optional
        #    Correlated solver.

        #Returns
        #-------
        #results : self.Results
        #"""
        #if bno_threshold is None:
        #    bno_threshold = self.opts.bno_threshold
        #if bno_threshold_occ is None:
        #    bno_threshold_occ = self.opts.bno_threshold_occ
        #if bno_threshold_vir is None:
        #    bno_threshold_vir = self.opts.bno_threshold_vir

        #bno_threshold = BNO_Threshold(self.opts.bno_truncation, bno_threshold)

        #if bno_threshold_occ is not None:
        #    bno_threshold_occ = BNO_Threshold(self.opts.bno_truncation, bno_threshold_occ)
        #if bno_threshold_vir is not None:
        #    bno_threshold_vir = BNO_Threshold(self.opts.bno_truncation, bno_threshold_vir)

        #if solver is None:
        #    solver = self.solver
        #if self.bath is None:
        #    self.make_bath()

        #cluster = self.make_cluster(self.bath, bno_threshold=bno_threshold,
        #        bno_threshold_occ=bno_threshold_occ, bno_threshold_vir=bno_threshold_vir)
        #cluster.log_sizes(self.log.info, header="Orbitals for %s with %s" % (self, bno_threshold))

        #if mpi:
        #    self.base.communicate_clusters()

    def kernel(self, solver=None, init_guess=None, eris=None):

        if solver is None:
            solver = self.solver
        if self.cluster is None:
            raise RuntimeError
        cluster = self.cluster

        # For self-consistent calculations, we can reuse ERIs:
        if eris is None:
            eris = self._eris
        #if (eris is not None) and (eris.mo_coeff.size > cluster.c_active.size):
        #    self.log.debugv("Projecting ERIs onto subspace")
        #    eris = ao2mo.helper.project_ccsd_eris(eris, cluster.c_active, cluster.nocc_active, ovlp=self.base.get_ovlp())

        if solver is None:
            return None

        init_guess = self.get_init_guess(init_guess, solver, cluster)

        # Create solver object
        solver_cls = get_solver_class(self.mf, solver)
        solver_opts = self.get_solver_options(solver)
        cluster_solver = solver_cls(self.mf, self, cluster, **solver_opts)

        # --- Chemical potential
        cpt_frag = self.base.opts.global_frag_chempot
        if self.opts.nelectron_target is not None:
            cluster_solver.optimize_cpt(self.opts.nelectron_target, c_frag=self.c_proj)
        elif cpt_frag:
            # Add chemical potential to fragment space
            r = self.get_overlap('cluster|frag')
            if self.base.is_rhf:
                p_frag = np.dot(r, r.T)
                cluster_solver.v_ext = cpt_frag * p_frag
            else:
                p_frag = (np.dot(r[0], r[0].T), np.dot(r[1], r[1].T))
                cluster_solver.v_ext = (cpt_frag * p_frag[0], cpt_frag * p_frag[1])

        # --- Coupled fragments
        if self.opts.coupled_iterations:
            if solver != 'CCSD':
                raise NotImplementedError()
            if not mpi or len(self.base.fragments) > len(mpi):
                raise RuntimeError("coupled_iterations requires MPI.")
            cluster_solver.couple_iterations(self.base.fragments)

        if eris is None:
            eris = cluster_solver.get_eris()
        if not self.base.opts._debug_exact_wf:
            with log_time(self.log.info, ("Time for %s solver:" % solver) + " %s"):
                cluster_solver.kernel(eris=eris, **init_guess)
        else:
            cluster_solver._debug_exact_wf(self.base._debug_exact_wf)

        # --- Add to results data class
        results = self._results
        results.n_active = cluster.norb_active
        results.converged = cluster_solver.converged
        # Unprojected WF
        results.wf = cluster_solver.wf
        # Projected WF
        proj = self.get_overlap('frag|cluster-occ')
        if isinstance(cluster_solver.wf, RFCI_WaveFunction):
            pwf = cluster_solver.wf.to_cisd()
        else:
            pwf = cluster_solver.wf
        results.pwf = pwf.project(proj, inplace=False)

        # Keep ERIs stored
        if (self.opts.store_eris or self.base.opts.store_eris):
            self._eris = eris

        return results

    def get_solver_options(self, solver):
        # TODO: fix this mess...
        solver_opts = {}
        solver_opts.update(self.opts.solver_options)
        #pass_through = ['make_rdm1', 'make_rdm2']
        pass_through = []
        if 'CCSD' in solver.upper():
            pass_through += ['solve_lambda', 't_as_lambda', 'sc_mode', 'dm_with_frozen']
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

    # --- Expectation values
    # ----------------------

    # --- Energies

    def get_energy_prefactor(self):
        return self.sym_factor * self.opts.energy_factor

    def get_fragment_energy(self, c1, c2, eris=None, fock=None, c2ba_order='ba', axis1='fragment'):
        """Calculate fragment correlation energy contribution from projected C1, C2.

        Parameters
        ----------
        c1: (n(occ-CO), n(vir-CO)) array
            Fragment projected C1-amplitudes.
        c2: (n(occ-CO), n(occ-CO), n(vir-CO), n(vir-CO)) array
            Fragment projected C2-amplitudes.
        eris: array or PySCF _ChemistERIs object
            Electron repulsion integrals as returned by ccsd.ao2mo().
        fock: (n(AO), n(AO)) array, optional
            Fock matrix in AO representation. If None, self.base.get_fock_for_energy()
            is used. Default: None.

        Returns
        -------
        e_singles: float
            Fragment correlation energy contribution from single excitations.
        e_doubles: float
            Fragment correlation energy contribution from double excitations.
        e_corr: float
            Total fragment correlation energy contribution.
        """
        if not self.get_energy_prefactor(): return (0, 0, 0)
        nocc, nvir = c2.shape[1:3]
        occ, vir = np.s_[:nocc], np.s_[nocc:]
        if axis1 == 'fragment':
            px = self.get_overlap('frag|cluster-occ')

        # --- Singles energy (zero for HF-reference)
        if c1 is not None:
            if fock is None:
                fock = self.base.get_fock_for_energy()
            fov =  dot(self.cluster.c_active_occ.T, fock, self.cluster.c_active_vir)
            if axis1 == 'fragment':
                e_singles = 2*einsum('ia,xi,xa->', fov, px, c1)
            else:
                e_singles = 2*np.sum(fov*c1)
        else:
            e_singles = 0
        # --- Doubles energy
        if eris is None:
            eris = self._eris
        if hasattr(eris, 'ovvo'):
            g_ovvo = eris.ovvo[:]
        elif hasattr(eris, 'ovov'):
            # MP2 only has eris.ovov - for real integrals we transpose
            g_ovvo = eris.ovov[:].reshape(nocc,nvir,nocc,nvir).transpose(0, 1, 3, 2).conj()
        elif eris.shape == (nocc, nvir, nocc, nvir):
            g_ovvo = eris.transpose(0,1,3,2)
        else:
            g_ovvo = eris[occ,vir,vir,occ]

        if axis1 == 'fragment':
            e_doubles = (2*einsum('xi,xjab,iabj', px, c2, g_ovvo)
                         - einsum('xi,xjab,ibaj', px, c2, g_ovvo))
        else:
            e_doubles = (2*einsum('ijab,iabj', c2, g_ovvo)
                         - einsum('ijab,ibaj', c2, g_ovvo))

        e_singles = (self.get_energy_prefactor() * e_singles)
        e_doubles = (self.get_energy_prefactor() * e_doubles)
        e_corr = (e_singles + e_doubles)
        return e_singles, e_doubles, e_corr


    # --- Density-matrices

    def _ccsd_amplitudes_for_dm(self, t_as_lambda=None, sym_t2=True):
        if t_as_lambda is None:
            t_as_lambda = self.opts.t_as_lambda
        wf = self.results.wf.as_ccsd()
        t1, t2 = wf.t1, wf.t2
        pwf = self.results.pwf.restore(sym=sym_t2).as_ccsd()
        t1x, t2x = pwf.t1, pwf.t2
        # Lambda amplitudes
        if t_as_lambda:
            l1, l2 = t1, t2
            l1x, l2x = t1x, t2x
        else:
            l1, l2 = wf.l1, wf.l2
            l1x, l2x = pwf.l1, pwf.l2
        return t1, t2, l1, l2, t1x, t2x, l1x, l2x

    def _get_projected_gamma1_intermediates(self, t_as_lambda=None, sym_t2=True):
        t1, t2, l1, l2, t1x, t2x, l1x, l2x = self._ccsd_amplitudes_for_dm(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
        doo, dov, dvo, dvv = pyscf.cc.ccsd_rdm._gamma1_intermediates(None, t1, t2, l1x, l2x)
        # Correction for term without Lambda amplitude:
        dvo += (t1x - t1).T
        d1 = (doo, dov, dvo, dvv)
        return d1

    def _get_projected_gamma2_intermediates(self, t_as_lambda=None, sym_t2=True):
        t1, t2, l1, l2, t1x, t2x, l1x, l2x = self._ccsd_amplitudes_for_dm(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
        cc = self.mf # Only attributes stdout, verbose, and max_memory are needed, just use mean-field object
        dovov, *d2rest = pyscf.cc.ccsd_rdm._gamma2_intermediates(cc, t1, t2, l1x, l2x)
        # Correct D2[ovov] part (first element of d2 tuple)
        dtau = ((t2x-t2) + einsum('ia,jb->ijab', (t1x-t1), t1))
        dovov += dtau.transpose(0,2,1,3)
        dovov -= dtau.transpose(0,3,1,2)/2
        d2 = (dovov, *d2rest)
        return d2

    def make_fragment_dm1(self, t_as_lambda=None, sym_t2=True):
        """Currently CCSD only.

        Without mean-field contribution!"""
        d1 = self._get_projected_gamma1_intermediates(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
        dm1 = pyscf.cc.ccsd_rdm._make_rdm1(None, d1, with_frozen=False, with_mf=False)
        return dm1

    def make_fragment_dm2cumulant(self, t_as_lambda=False, sym_t2=True, sym_dm2=True, full_shape=True,
            approx_cumulant=True):
        """Currently MP2/CCSD only"""

        if self.solver == 'MP2':
            if approx_cumulant not in (1, True):
                raise NotImplementedError
            t2x = self.results.pwf.restore(sym=sym_t2).as_ccsd().t2
            dovov = 2*(2*t2x - t2x.transpose(0,1,3,2)).transpose(0,2,1,3)
            if not full_shape:
                return dovov
            nocc, nvir = dovov.shape[:2]
            norb = nocc+nvir
            dm2 = np.zeros(4*[norb])
            occ, vir = np.s_[:nocc], np.s_[nocc:]
            dm2[occ,vir,occ,vir] = dovov
            dm2[vir,occ,vir,occ] = dovov.transpose(1,0,3,2)
            return dm2

        cc = d1 = None
        d2 = self._get_projected_gamma2_intermediates(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
        dm2 = pyscf.cc.ccsd_rdm._make_rdm2(cc, d1, d2, with_dm1=False, with_frozen=False)
        if (approx_cumulant == 2):
            raise NotImplementedError
        elif (approx_cumulant in (1, True)):
            pass
        elif not approx_cumulant:
            # Remove dm1(cc)^2
            dm1x = self.make_fragment_dm1(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
            dm1 = self.results.wf.make_rdm1(with_mf=False)
            dm2 -= (einsum('ij,kl->ijkl', dm1, dm1x)/2 + einsum('ij,kl->ijkl', dm1x, dm1)/2
                  - einsum('ij,kl->iklj', dm1, dm1x)/4 - einsum('ij,kl->iklj', dm1x, dm1)/4)

        if (sym_dm2 and not sym_t2):
            dm2 = (dm2 + dm2.transpose(1,0,3,2) + dm2.transpose(2,3,0,1) + dm2.transpose(3,2,1,0))/4
        return dm2

    #def make_partial_dm1_energy(self, t_as_lambda=False):
    #    dm1 = self.make_partial_dm1(t_as_lambda=t_as_lambda)
    #    c_act = self.cluster.c_active
    #    fock = np.linalg.multi_dot((c_act.T, self.base.get_fock(), c_act))
    #    e_dm1 = einsum('ij,ji->', fock, dm1)
    #    return e_dm1

    def make_fragment_dm2cumulant_energy(self, t_as_lambda=False, sym_t2=True, approx_cumulant=True):
        dm2 = self.make_fragment_dm2cumulant(t_as_lambda=t_as_lambda, sym_t2=sym_t2, approx_cumulant=approx_cumulant,
                full_shape=False)
        fac = (2 if self.solver == 'MP2' else 1)
        if self._eris is None:
            eris = self.base.get_eris_array(self.cluster.c_active)
        # CCSD
        elif hasattr(self._eris, 'ovoo'):
            eris = vayesta.core.ao2mo.helper.get_full_array(self._eris)
        # MP2
        else:
            eris = self._eris
        e_dm2 = fac*einsum('ijkl,ijkl->', eris, dm2)/2
        return e_dm2

    # --- Other
    # ---------

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
