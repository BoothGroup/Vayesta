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
from vayesta.solver import get_solver_class2 as get_solver_class
from vayesta.core.fragmentation import IAO_Fragmentation

from vayesta.core.bath import DMET_Bath, BNO_Bath, CompleteBath
from vayesta.core.actspace import ActiveSpace

from . import ewf
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

        self.cluster = None

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

    def reset(self):
        super().reset()

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
        if bath_type is NotSet:
            bath_type = self.opts.bath_type
        # DMET bath only
        if bath_type is None or bath_type.lower() == 'dmet':
            bath = DMET_Bath(self, dmet_threshold=self.opts.dmet_threshold)
        # All environment orbitals as bath
        elif bath_type.lower() in ('all', 'full'):
            bath = CompleteBath(self, dmet_threshold=self.opts.dmet_threshold)
        # MP2 bath natural orbitals
        elif bath_type.lower() == 'mp2-bno':
            bath = BNO_Bath(self, dmet_threshold=self.opts.dmet_threshold)
        else:
            raise ValueError("Unknown bath_type: %r" % bath_type)
        bath.kernel()
        self.bath = bath
        return bath

    def make_cluster(self, bath, bno_threshold=None, bno_number=None):

        c_bno_occ, c_frozen_occ = bath.get_occupied_bath(bno_threshold[0], bno_number[0])
        c_bno_vir, c_frozen_vir = bath.get_virtual_bath(bno_threshold[1], bno_number[1])

        # Canonicalize orbitals
        c_active_occ = self.canonicalize_mo(bath.c_cluster_occ, c_bno_occ)[0]
        c_active_vir = self.canonicalize_mo(bath.c_cluster_vir, c_bno_vir)[0]
        # Do not overwrite self.c_active_occ/vir yet - we still need the previous coefficients
        # to generate an intial guess

        cluster = ActiveSpace(self.mf, c_active_occ, c_active_vir, c_frozen_occ=c_frozen_occ, c_frozen_vir=c_frozen_vir)

        # Check occupations
        self.check_mo_occupation((2 if self.base.is_rhf else 1), cluster.c_occ)
        self.check_mo_occupation(0, cluster.c_vir)
        self.cluster = cluster
        return cluster

    def get_init_guess(self, init_guess, solver, cluster):
        # TODO: clean
        # --- Project initial guess and integrals from previous cluster calculation with smaller eta:
        # Use initial guess from previous calculations
        # For self-consistent calculations, we can restart calculation:
        if init_guess is None and 'ccsd' in solver.lower():
            if self.base.opts.sc_mode and self.base.iteration > 1:
                self.log.debugv("Restarting using T1,T2 from previous iteration")
                init_guess = {'t1' : self.results.t1, 't2' : self.results.t2}
            elif self.base.opts.project_init_guess and self.results is not None:
                self.log.debugv("Restarting using projected previous T1,T2")
                # Projectors for occupied and virtual orbitals
                p_occ = dot(self.c_active_occ.T, self.base.get_ovlp(), cluster.c_active_occ)
                p_vir = dot(self.c_active_vir.T, self.base.get_ovlp(), cluster.c_active_vir)
                #t1, t2 = init_guess.pop('t1'), init_guess.pop('t2')
                t1, t2 = helper.transform_amplitudes(self.results.t1, self.results.t2, p_occ, p_vir)
                init_guess = {'t1' : t1, 't2' : t2}
        if init_guess is None: init_guess = {}
        return init_guess

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

        cluster = self.make_cluster(self.bath, bno_threshold=bno_threshold, bno_number=bno_number)
        cluster.log_sizes(self.log.info, header="Orbitals for %s" % self)

        init_guess = self.get_init_guess(init_guess, solver, cluster)

        # For self-consistent calculations, we can reuse ERIs:
        if eris is None:
            if self.base.opts.sc_mode and self.base.iteration > 1:
                self.log.debugv("Reusing ERIs from previous iteration")
                eris = self.results.eris
            # If superspace ERIs were calculated before, they can be transformed and used again:
            elif self.base.opts.project_eris and self.results is not None:
                t0 = timer()
                self.log.debugv("Projecting previous ERIs onto subspace")
                eris = psubspace.project_eris(self.results.eris, cluster.c_active_occ, cluster.c_active_vir, ovlp=self.base.get_ovlp())
                self.log.timingv("Time to project ERIs:  %s", time_string(timer()-t0))

        # We can now overwrite the orbitals from last BNO run:
        self._c_active_occ = cluster.c_active_occ
        self._c_active_vir = cluster.c_active_vir

        if solver is None:
            return None

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
        # C1 and C2 are in intermediate normalization:
        p1 = self.project_amplitude_to_fragment(cluster_solver.get_c1(), cluster.c_active_occ, cluster.c_active_vir)
        p2 = self.project_amplitude_to_fragment(cluster_solver.get_c2(), cluster.c_active_occ, cluster.c_active_vir)

        e_corr = self.get_fragment_energy(p1, p2, eris=eris)
        if bno_threshold[0] is not None:
            if bno_threshold[0] == bno_threshold[1]:
                self.log.info("BNO threshold= %.1e :  E(corr)= %+14.8f Ha", bno_threshold[0], e_corr)
            else:
                self.log.info("BNO threshold= %.1e / %.1e :  E(corr)= %+14.8f Ha", *bno_threshold, e_corr)
        else:
            self.log.info("BNO number= %3d / %3d:  E(corr)= %+14.8f Ha", *bno_number, e_corr)

        results = self.Results(fid=self.id, bno_threshold=bno_threshold, n_active=cluster.norb_active,
                converged=cluster_solver.converged, e_corr=e_corr)
        if self.opts.make_rdm1:
            results.dm1 = cluster_solver.make_rdm1()
        if self.opts.make_rdm2:
            results.dm2 = cluster_solver.make_rdm2()

        #(results.t1_pf, results.t2_pf), (results.e1b, results.e2b_conn, results.e2b_disc) = self.project_solver_results(solver_results)

        # Keep Amplitudes [optional]
        if self.base.opts.project_init_guess or self.opts.sc_mode:
            if hasattr(cluster_solver, 't2'):
                results.t1 = cluster_solver.t1
                results.t2 = cluster_solver.t2
            if hasattr(cluster_solver, 'c2'):
                results.c0 = cluster_solver.c0
                results.c1 = cluster_solver.c1
                results.c2 = cluster_solver.c2
        # Keep Lambda-Amplitudes
        if hasattr(cluster_solver, 'l2') and cluster_solver.l2 is not None:
            results.l1 = cluster_solver.l1
            results.l2 = cluster_solver.l2
        # Keep ERIs [optional]
        if self.base.opts.project_eris or self.opts.sc_mode:
            results.eris = eris

        self._results = results

        # DMET energy
        calc_dmet = self.opts.calculate_e_dmet
        if calc_dmet == 'auto':
            calc_dmet = (results.dm1 is not None and results.dm2 is not None)
        if calc_dmet:
            results.e_dmet = self.get_fragment_dmet_energy(dm1=results.dm1, dm2=results.dm2, eris=eris)

        # Force GC to free memory
        if False:
            m0 = get_used_memory()
            del cluster_solver#, solver_results
            ndel = gc.collect()
            self.log.debugv("GC deleted %d objects and freed %.3f MB of memory", ndel, (get_used_memory()-m0)/1e6)

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
        if self.opts.energy_factor == 0: return 0

        nocc, nvir = p2.shape[1:3]
        occ = np.s_[:nocc]
        vir = np.s_[nocc:]
        # Singles energy (for non-HF reference)
        if p1 is not None:
            if hasattr(eris, 'fock'):
                f = eris.fock[occ,vir]
            else:
                f = dot(self.c_active_occ.T, self.base.get_fock(), self.c_active_vir)
            e_s = 2*np.sum(f * p1)
        else:
            e_s = 0
        # E2
        if hasattr(eris, 'ovvo'):
            g_ovvo = eris.ovvo[:]
        elif hasattr(eris, 'ovov'):
            # MP2 only has eris.ovov - for real integrals we transpose
            g_ovvo = eris.ovov[:].reshape(nocc,nvir,nocc,nvir).transpose(0, 1, 3, 2).conj()
        else:
            g_ovvo = eris[occ,vir,vir,occ]

        e_d = 2*einsum('ijab,iabj', p2, g_ovvo) - einsum('ijab,jabi', p2, g_ovvo)
        self.log.debug("Energy components: E(singles)= %s, E(doubles)= %s",
                energy_string(e_s), energy_string(e_d))
        if e_s > 0.1*e_d and e_s > 1e-4:
            self.log.warning("Large E(singles) component!")
        e_frag = self.opts.energy_factor * self.sym_factor * (e_s + e_d)
        return e_frag

    def pop_analysis(self, cluster=None, dm1=None, **kwargs):
        if cluster is None: cluster = self.cluster
        if dm1 is None: dm1 = self.results.dm1
        if dm1 is None: raise ValueError()
        # Add frozen mean-field contribution:
        dm1 = cluster.add_frozen_rdm1(dm1)
        return self.base.pop_analysis(dm1, mo_coeff=cluster.coeff, **kwargs)

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
