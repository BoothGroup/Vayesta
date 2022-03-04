# Standard libaries
from datetime import datetime
import dataclasses
from typing import Union

# External libaries
import numpy as np

# Internal libaries
import pyscf
import pyscf.pbc

# Local modules
from vayesta.core.util import *
from vayesta.core import Fragment
from vayesta.solver import get_solver_class2 as get_solver_class
from vayesta.core.fragmentation import IAO_Fragmentation

from vayesta.core.bath import BNO_Threshold
from vayesta.core.bath import DMET_Bath
from vayesta.core.bath import EwDMET_Bath
from vayesta.core.bath import BNO_Bath
from vayesta.core.bath import MP2_BNO_Bath
from vayesta.core.bath import CompleteBath
from vayesta.core.actspace import ActiveSpace
from vayesta.core import ao2mo

from . import ewf

# Get MPI rank of fragment
get_fragment_mpi_rank = lambda *args : args[0].mpi_rank

class EWFFragment(Fragment):

    @dataclasses.dataclass
    class Options(Fragment.Options):
        """Attributes set to `NotSet` inherit their value from the parent EWF object."""
        # Options also present in `base`:
        dmet_threshold: float = NotSet
        make_rdm1: bool = NotSet
        make_rdm2: bool = NotSet
        #solve_lambda: bool = NotSet                 # If False, use T-amplitudes inplace of Lambda-amplitudes
        t_as_lambda: bool = NotSet                  # If True, use T-amplitudes inplace of Lambda-amplitudes
        eom_ccsd: list = NotSet
        eom_ccsd_nroots: int = NotSet
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
        #
        # --- Energy
        calculate_e_dmet: bool = 'auto'
        e_corr_part: str = NotSet
        #
        dm_with_frozen: bool = NotSet
        # --- Solver options
        tcc_fci_opts: dict = dataclasses.field(default_factory=dict)
        # --- Intercluster MP2 energy
        icmp2_bno_threshold: float = NotSet
        # --- Storage
        store_t1:  Union[bool,str] = NotSet
        store_t2:  Union[bool,str] = NotSet
        store_l1:  Union[bool,str] = NotSet
        store_l2:  Union[bool,str] = NotSet
        store_t1x: Union[bool,str] = NotSet
        store_t2x: Union[bool,str] = NotSet
        store_l1x: Union[bool,str] = NotSet
        store_l2x: Union[bool,str] = NotSet
        store_dm1: Union[bool,str] = NotSet
        store_dm2: Union[bool,str] = NotSet


    @dataclasses.dataclass
    class Results(Fragment.Results):
        bno_threshold: float = None
        n_active: int = None
        ip_energy: np.ndarray = None
        ea_energy: np.ndarray = None

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
        if solver not in ewf.VALID_SOLVERS:
            raise ValueError("Unknown solver: %s" % solver)
        self.solver = solver

        # For self-consistent mode
        self.solver_results = None

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
        cluster = ActiveSpace(self.mf, c_active_occ, c_active_vir, c_frozen_occ=c_frozen_occ, c_frozen_vir=c_frozen_vir)

        def check_occupation(mo_coeff, expected):
            occup = self.get_mo_occupation(mo_coeff)
            # RHF
            if np.ndim(occup[0]) == 0:
                assert np.allclose(occup, 2*expected, rtol=0, atol=2*self.opts.dmet_threshold)
            else:
                assert np.allclose(occup[0], expected, rtol=0, atol=self.opts.dmet_threshold)
                assert np.allclose(occup[1], expected, rtol=0, atol=self.opts.dmet_threshold)

        check_occupation(cluster.c_occ, 1)
        check_occupation(cluster.c_vir, 0)

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

    def kernel(self, bno_threshold=None, bno_threshold_occ=None, bno_threshold_vir=None, solver=None, init_guess=None, eris=None):
        """Run solver for a single BNO threshold.

        Parameters
        ----------
        bno_threshold : float, optional
            Bath natural orbital (BNO) threshold.
        solver : {'MP2', 'CISD', 'CCSD', 'FCI'}, optional
            Correlated solver.

        Returns
        -------
        results : self.Results
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

        if solver is None:
            solver = self.solver
        if self.bath is None:
            self.make_bath()

        cluster = self.make_cluster(self.bath, bno_threshold=bno_threshold,
                bno_threshold_occ=bno_threshold_occ, bno_threshold_vir=bno_threshold_vir)
        cluster.log_sizes(self.log.info, header="Orbitals for %s with %s" % (self, bno_threshold))

        # For self-consistent calculations, we can reuse ERIs:
        if eris is None:
            eris = self._eris
        if (eris is not None) and (eris.mo_coeff.shape != cluster.c_active.shape):
            self.log.debugv("Projecting ERIs onto subspace")
            eris = ao2mo.helper.project_ccsd_eris(eris, cluster.c_active, cluster.nocc_active, ovlp=self.base.get_ovlp())

        # We can now overwrite the orbitals from last BNO run:
        #self._c_active_occ = cluster.c_active_occ
        #self._c_active_vir = cluster.c_active_vir

        if solver is None:
            return None

        init_guess = self.get_init_guess(init_guess, solver, cluster)

        # Create solver object
        solver_cls = get_solver_class(self.mf, solver)
        solver_opts = self.get_solver_options(solver)
        # OLD CALL:
        #cluster_solver = solver_cls(self, mo_coeff, mo_occ, nocc_frozen=cluster.nocc_frozen, nvir_frozen=cluster.nvir_frozen, **solver_opts)
        # NEW CALL:
        cluster_solver = solver_cls(self, cluster, **solver_opts)
        if self.opts.nelectron_target is not None:
            cluster_solver.optimize_cpt(self.opts.nelectron_target, c_frag=self.c_proj)
        if eris is None:
            eris = cluster_solver.get_eris()
        with log_time(self.log.info, ("Time for %s solver:" % solver) + " %s"):
            cluster_solver.kernel(eris=eris, **init_guess)

        # Get projected amplitudes ('p1', 'p2')
        if hasattr(cluster_solver, 'c0'):
            self.log.info("Weight of reference determinant= %.8g", abs(cluster_solver.c0))
        # --- Calculate energy
        with log_time(self.log.info, ("Time for fragment energy= %s")):
            c1x = self.project_amp1_to_fragment(cluster_solver.get_c1(intermed_norm=True))
            c2x = self.project_amp2_to_fragment(cluster_solver.get_c2(intermed_norm=True))
            e_singles, e_doubles, e_corr = self.get_fragment_energy(c1x, c2x, eris=eris)
            del c1x, c2x

        if (solver != 'FCI' and (e_singles > max(0.1*e_doubles, 1e-4))):
            self.log.warning("Large singles energy component: E(S)= %s, E(D)= %s",
                    energy_string(e_singles), energy_string(e_doubles))
        else:
            self.log.debug("Energy components: E(S)= %s, E(D)= %s", energy_string(e_singles), energy_string(e_doubles))
        self.log.info("%s:  E(corr)= %+14.8f Ha", bno_threshold, e_corr)

        # --- Add to results
        results = self._results
        results.bno_threshold = bno_threshold
        results.n_active = cluster.norb_active
        results.converged = cluster_solver.converged
        results.e_corr = e_corr

        # Store density-matrix
        if self.opts.store_dm1 is True or self.opts.make_rdm1:
            results.dm1 = cluster_solver.make_rdm1()
        if self.opts.store_dm2 is True or self.opts.make_rdm2:
            results.dm2 = cluster_solver.make_rdm2()
        # Store wave function amplitudes
        if self.opts.store_t1:
            results.t1 = cluster_solver.get_t1()
        if self.opts.store_t2:
            results.t2 = cluster_solver.get_t2()
        solve_lambda = np.any([(getattr(self.opts, 'store_%s' % s) is True) for s in ['l1', 'l2', 'l1x', 'l2x']])
        if self.opts.store_l1 and hasattr(cluster_solver, 'get_l1'):
            l1 = cluster_solver.get_l1(solve_lambda=solve_lambda)
            if l1 is not None:
                results.l1 = l1
        if self.opts.store_l2 and hasattr(cluster_solver, 'get_l2'):
            l2 = cluster_solver.get_l2(solve_lambda=solve_lambda)
            if l2 is not None:
                results.l2 = l2
        if self.opts.store_t1x:
            results.t1x = self.project_amp1_to_fragment(cluster_solver.get_t1())
        if self.opts.store_t2x:
            results.t2x = self.project_amp2_to_fragment(cluster_solver.get_t2())
        if self.opts.store_l1x and hasattr(cluster_solver, 'get_l1'):
            l1 = cluster_solver.get_l1(solve_lambda=solve_lambda)
            if l1 is not None:
                results.l1x = self.project_amp1_to_fragment(l1)
        if self.opts.store_l2x and hasattr(cluster_solver, 'get_l2'):
            l2 = cluster_solver.get_l2(solve_lambda=solve_lambda)
            if l2 is not None:
                results.l2x = self.project_amp2_to_fragment(l2)
        self._results = results

        # DMET energy
        calc_dmet = self.opts.calculate_e_dmet
        if calc_dmet == 'auto':
            calc_dmet = (results.dm1 is not None and results.dm2 is not None)
        if calc_dmet:
            results.e_dmet = self.get_fragment_dmet_energy(dm1=results.dm1, dm2=results.dm2, eris=eris)

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
            pass_through += ['t_as_lambda', 'sc_mode', 'dm_with_frozen', 'eom_ccsd', 'eom_ccsd_nroots']
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

    def get_energy_prefactor(self):
        return self.sym_factor * self.opts.energy_factor

    def get_fragment_energy(self, c1, c2, eris, fock=None, axis1='fragment'):
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
            px = self.get_occ2frag_projector()

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
        if hasattr(eris, 'ovvo'):
            g_ovvo = eris.ovvo[:]
        elif hasattr(eris, 'ovov'):
            # MP2 only has eris.ovov - for real integrals we transpose
            g_ovvo = eris.ovov[:].reshape(nocc,nvir,nocc,nvir).transpose(0, 1, 3, 2).conj()
        elif eris.shape == (nocc, nvir, nocc, nvir):
            g_ovvo = eris.transpose(0,1,3,2)
        else:
            g_ovvo = eris[occ,vir,vir,occ]

        e_doubles = 0
        if axis1 == 'fragment':
            if self.opts.e_corr_part in ('all', 'direct'):
                e_doubles += 2*einsum('xi,xjab,iabj', px, c2, g_ovvo)
            if self.opts.e_corr_part in ('all', 'exchange'):
                e_doubles -= einsum('xi,xjab,jabi', px, c2, g_ovvo)
        else:
            if self.opts.e_corr_part in ('all', 'direct'):
                e_doubles += 2*einsum('ijab,iabj', c2, g_ovvo)
            if self.opts.e_corr_part in ('all', 'exchange'):
                e_doubles -= einsum('ijab,jabi', c2, g_ovvo)

        e_singles = (self.get_energy_prefactor() * e_singles)
        e_doubles = (self.get_energy_prefactor() * e_doubles)
        e_corr = (e_singles + e_doubles)
        return e_singles, e_doubles, e_corr

    def get_cluster_sz(self, proj=None):
        return 0.0

    def get_cluster_ssz(self, proj1=None, proj2=None, dm1=None, dm2=None):
        """<P(A) S_z P(B) S_z>"""
        dm1 = (self.results.dm1 if dm1 is None else dm1)
        dm2 = (self.results.dm2 if dm2 is None else dm2)
        if (dm1 is None or dm2 is None):
            raise ValueError()
        dm1a = dm1/2
        dm2aa = (dm2 - dm2.transpose(0,3,2,1)) / 6
        dm2ab = (dm2/2 - dm2aa)

        if proj1 is None:
            ssz = (einsum('iijj->', dm2aa) - einsum('iijj->', dm2ab))/2
            ssz += einsum('ii->', dm1a)/2
            return ssz
        if proj2 is None:
            proj2 = proj1
        ssz = (einsum('ijkl,ij,kl->', dm2aa, proj1, proj2)
             - einsum('ijkl,ij,kl->', dm2ab, proj1, proj2))/2
        ssz += einsum('ij,ik,jk->', dm1a, proj1, proj2)/2
        return ssz

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
