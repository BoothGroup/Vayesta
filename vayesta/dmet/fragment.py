import dataclasses
from timeit import default_timer as timer

# External libaries
import numpy as np

from vayesta.core import Fragment
from vayesta.core.bath import BNO_Threshold
from vayesta.solver import get_solver_class
from vayesta.core.util import *

import vayesta.ewf

# We might want to move the useful things from here into core, since they seem pretty general.


class DMETFragmentExit(Exception):
    pass

VALID_SOLVERS = [None, "", "MP2", "CISD", "CCSD", "CCSD(T)", 'FCI', "FCI-spin0", "FCI-spin1"]


class DMETFragment(Fragment):

    @dataclasses.dataclass
    class Options(Fragment.Options):
        """Attributes set to `NotSet` inherit their value from the parent DMET object."""
        # Options also present in `base`:
        dmet_threshold: float = NotSet
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
    class Results(Fragment.Results):
        fid: int = None
        bno_threshold: float = None
        n_active: int = None
        converged: bool = None
        # For DM1:
        g1: np.ndarray = None
        dm1: np.ndarray = None
        dm2: np.ndarray = None
        # energy contributions.
        e1: float = None
        e2: float = None

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

    make_bath = vayesta.ewf.fragment.Fragment.make_bath

    make_cluster = vayesta.ewf.fragment.Fragment.make_cluster

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

        if solver is None:
            solver = self.solver
        if self.bath is None or construct_bath:
            self.make_bath()

        if bno_number is not None:
            bno_threshold = BNO_Threshold('number', bno_number)
        else:
            bno_threshold = BNO_Threshold('occupation', bno_threshold)

        cluster = self.make_cluster(self.bath, bno_threshold=bno_threshold)
        self.cluster = cluster
        self.log.info('Orbitals for %s', self)
        self.log.info('-------------%s', len(str(self))*'-')
        self.log.info(cluster.repr_size().replace('%', '%%'))
        # If we want to reuse previous info for initial guess and eris we'd do that here...
        # We can now overwrite the orbitals from last BNO run:
        self._c_active_occ = cluster.c_active_occ
        self._c_active_vir = cluster.c_active_vir

        if solver is None:
            return None

        # Create solver object
        solver_cls = get_solver_class(self.mf, solver)
        solver_opts = self.get_solver_options(solver)
        cluster_solver = solver_cls(self.mf, self, cluster, **solver_opts)
        # Chemical potential
        if chempot is not None:
            cluster_solver.v_ext = -chempot * self.get_fragment_projector(self.cluster.c_active)
        if eris is None:
            eris = cluster_solver.get_eris()
        with log_time(self.log.info, ("Time for %s solver:" % solver) + " %s"):
            cluster_solver.kernel(eris=eris)

        results = self._results

        results.bno_threshold = bno_threshold
        results.n_active = self.cluster.norb_active
        # Need to rewrite EBFCI solver to expose this properly...
        results.converged = True

        results.dm1 = cluster_solver.make_rdm1()
        results.dm2 = cluster_solver.make_rdm2()
        results.e1, results.e2 = self.get_dmet_energy_contrib()

        return results

    def get_solver_options(self, solver):
        solver_opts = {}
        solver_opts.update(self.opts.solver_options)
        # pass_through = ['make_rdm1', 'make_rdm2']
        pass_through = []
        # if 'CCSD' in solver.upper():
        #    pass_through += ['dm_with_frozen', 'eom_ccsd', 'eom_ccsd_nroots']
        for attr in pass_through:
            self.log.debugv("Passing fragment option %s to solver.", attr)
            solver_opts[attr] = getattr(self.opts, attr)

        return solver_opts

    def get_dmet_energy_contrib(self, eris=None):
        """Calculate the contribution of this fragment to the overall DMET energy."""
        # Projector to the impurity in the active basis.
        P_imp = self.get_fragment_projector(self.cluster.c_active)
        c_act = self.cluster.c_active
        if eris is None:
            with log_time(self.log.timing, "Time for AO->MO transformation: %s"):
                eris = self.base.get_eris_array(c_act)
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

    def get_energy_prefactor(self):
        # Defined for compatibility..
        return 1.0
