import dataclasses
from timeit import default_timer as timer

# External libaries
import numpy as np

from vayesta.core import QEmbeddingFragment
from vayesta.core.actspace import ActiveSpace
from vayesta.core.bath import DMET_Bath, BNO_Bath, MP2_BNO_Bath, CompleteBath
# Local modules
from vayesta.core.util import *
from vayesta.solver import get_solver_class2 as get_solver_class


# Internal libaries
# import pyscf.pbc
# We might want to move the useful things from here into core, since they seem pretty general.


class DMETFragmentExit(Exception):
    pass


VALID_SOLVERS = [None, "", "MP2", "CISD", "CCSD", "CCSD(T)", 'FCI', "FCI-spin0", "FCI-spin1"]


class DMETFragment(QEmbeddingFragment):
    @dataclasses.dataclass
    class Options(QEmbeddingFragment.Options):
        """Attributes set to `NotSet` inherit their value from the parent DMET object."""
        # Options also present in `base`:
        dmet_threshold: float = NotSet
        make_rdm1: bool = True
        make_rdm2: bool = True
        energy_factor: float = 1.0
        eom_ccsd: bool = NotSet
        energy_partitioning: str = NotSet
        sc_mode: int = NotSet
        # Bath type
        bath_type: str = NotSet
        bno_number: int = None  # Set a fixed number of BNOs
        # Additional fragment specific options:
        bno_threshold_factor: float = 1.0
        # CAS methods
        c_cas_occ: np.ndarray = None
        c_cas_vir: np.ndarray = None

    @dataclasses.dataclass
    class Results(QEmbeddingFragment.Results):
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
        dm2: np.ndarray = None

    def __init__(self, *args, solver=None, **kwargs):

        """
        Parameters
        ----------
        base : DMET
            Base DMET object.
        fid : int
            Unique ID of fragment.
        name :
            Name of fragment.
        """

        super().__init__(*args, **kwargs)

        if solver is None:
            solver = self.base.solver
        self.check_solver(solver)
        self.solver = solver
        self.solver_results = None

    def check_solver(self, solver):
        if solver not in VALID_SOLVERS:
            raise ValueError("Unknown solver: %s" % solver)

    @property
    def c_cluster_occ(self):
        return self.bath.c_cluster_occ

    @property
    def c_cluster_vir(self):
        return self.bath.c_cluster_vir

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
            bath = MP2_BNO_Bath(self, dmet_threshold=self.opts.dmet_threshold)
        else:
            raise ValueError("Unknown bath_type: %r" % bath_type)
        bath.kernel()
        self.bath = bath
        return bath

    def make_cluster(self, bath=None, bno_threshold=None, bno_number=None):
        if bath is None:
            bath = self.bath
        if bath is None:
            raise ValueError("make_cluster requires bath.")

        if isinstance(bath, BNO_Bath):
            c_bno_occ, c_frozen_occ = bath.get_occupied_bath(bno_threshold[0], bno_number[0])
            c_bno_vir, c_frozen_vir = bath.get_virtual_bath(bno_threshold[1], bno_number[1])
        else:
            c_bno_occ, c_frozen_occ = bath.get_occupied_bath()
            c_bno_vir, c_frozen_vir = bath.get_virtual_bath()

        # Canonicalize orbitals
        c_active_occ = self.canonicalize_mo(bath.c_cluster_occ, c_bno_occ)[0]
        c_active_vir = self.canonicalize_mo(bath.c_cluster_vir, c_bno_vir)[0]
        # Do not overwrite self.cluster.c_active_occ/vir yet - we still need the previous coefficients
        # to generate an intial guess
        cluster = ActiveSpace(self.mf, c_active_occ, c_active_vir, c_frozen_occ=c_frozen_occ, c_frozen_vir=c_frozen_vir)

        # Check occupations
        # self.check_mo_occupation((2 if self.base.is_rhf else 1), cluster.c_occ)
        # self.check_mo_occupation(0, cluster.c_vir)

        def check_occupation(mo_coeff, expected):
            occup = self.get_mo_occupation(mo_coeff)
            # RHF
            if np.ndim(occup[0]) == 0:
                assert np.allclose(occup, 2 * expected, rtol=0, atol=2 * self.opts.dmet_threshold)
            else:
                assert np.allclose(occup[0], expected, rtol=0, atol=self.opts.dmet_threshold)
                assert np.allclose(occup[1], expected, rtol=0, atol=self.opts.dmet_threshold)

        check_occupation(cluster.c_occ, 1)
        check_occupation(cluster.c_vir, 0)

        self.cluster = cluster
        return cluster

    def kernel(self, bno_threshold=None, bno_number=None, solver=None, init_guess=None, eris=None, construct_bath=True,
               chempot=None):
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
        if bno_number is None:
            bno_number = self.opts.bno_number
        if bno_number is None and bno_threshold is None:
            bno_threshold = self.base.bno_threshold
        if np.ndim(bno_threshold) == 0:
            bno_threshold = 2 * [bno_threshold]
        if np.ndim(bno_number) == 0:
            bno_number = 2 * [bno_number]

        if solver is None:
            solver = self.solver
        if self.bath is None or construct_bath:
            self.make_bath()

        cluster = self.make_cluster(self.bath, bno_threshold=bno_threshold, bno_number=bno_number)
        cluster.log_sizes(self.log.info, header="Orbitals for %s" % self)
        # If we want to reuse previous info for initial guess and eris we'd do that here...
        # We can now overwrite the orbitals from last BNO run:
        self._c_active_occ = cluster.c_active_occ
        self._c_active_vir = cluster.c_active_vir

        if solver is None:
            return None

        # Create solver object
        solver_cls = get_solver_class(self.mf, solver)
        solver_opts = self.get_solver_options(solver, chempot)
        cluster_solver = solver_cls(self, cluster, **solver_opts)
        if eris is None:
            eris = cluster_solver.get_eris()
        with log_time(self.log.info, ("Time for %s solver:" % solver) + " %s"):
            cluster_solver.kernel(eris=eris)

        results = self.Results(fid=self.id, bno_threshold=bno_threshold, n_active=cluster.norb_active,
                               converged=cluster_solver.converged, dm1=cluster_solver.make_rdm1(),
                               dm2=cluster_solver.make_rdm2())
        self._results = results

        return results

    def get_solver_options(self, solver, chempot=None):
        solver_opts = {}
        solver_opts.update(self.opts.solver_options)
        # pass_through = ['make_rdm1', 'make_rdm2']
        pass_through = []
        # if 'CCSD' in solver.upper():
        #    pass_through += ['dm_with_frozen', 'eom_ccsd', 'eom_ccsd_nroots']
        for attr in pass_through:
            self.log.debugv("Passing fragment option %s to solver.", attr)
            solver_opts[attr] = getattr(self.opts, attr)

        solver_opts["v_ext"] = None if chempot is None else - chempot * np.array(self.get_fragment_projector(
            self.cluster.c_active))

        return solver_opts

    def get_dmet_energy_contrib(self):
        """Calculate the contribution of this fragment to the overall DMET energy."""
        # Projector to the impurity in the active basis.
        P_imp = self.get_fragment_projector(self.cluster.c_active)
        c_act = self.cluster.c_active

        # Temporary implementation
        t0 = timer()
        eris = self.base.get_eris_array(c_act)
        self.log.timing("Time for AO->MO of (ij|kl):  %s", time_string(timer() - t0))

        nocc = self.cluster.c_active_occ.shape[1]
        occ = np.s_[:nocc]
        # Calculate the effective onebody interaction within the cluster.
        f_act = np.linalg.multi_dot((c_act.T, self.mf.get_fock(), c_act))
        v_act = 2 * np.einsum('iipq->pq', eris[occ, occ]) - np.einsum('iqpi->pq', eris[occ, :, :, occ])
        h_eff = f_act - v_act
        h_bare = np.linalg.multi_dot((c_act.T, self.base.get_hcore(), c_act))

        e1 = 0.5 * dot(P_imp, h_bare + h_eff, self.results.dm1).trace()
        e2 = 0.5 * np.einsum('pt,tqrs,pqrs->', P_imp, eris, self.results.dm2)
        # Code to generate the HF energy contribution for testing purposes.
        # mf_dm1 = np.linalg.multi_dot((c_act.T, self.base.get_ovlp(), self.mf.make_rdm1(),\
        #                               self.base.get_ovlp(), c_act))
        # e_hf = np.linalg.multi_dot((P_imp, 0.5 * (h_bare + f_act), mf_dm1)).trace()
        return e1, e2

    def get_frag_hl_dm(self):
        c = dot(self.c_frag.T, self.mf.get_ovlp(), self.cluster.c_active)
        return dot(c, self.results.dm1, c.T)

    def get_nelectron_hl(self):
        return self.get_frag_hl_dm().trace()
