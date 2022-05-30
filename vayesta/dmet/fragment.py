import dataclasses
from timeit import default_timer as timer

# External libaries
import numpy as np

from vayesta.core.qemb import Fragment
from vayesta.core.bath import BNO_Threshold
from vayesta.solver import get_solver_class
from vayesta.core.util import *


# We might want to move the useful things from here into core, since they seem pretty general.

class DMETFragmentExit(Exception):
    pass

@dataclasses.dataclass
class Results(Fragment.Results):
    n_active: int = None
    e1: float = None
    e2: float = None
    dm1: np.ndarray = None
    dm2: np.ndarray = None

class DMETFragment(Fragment):

    Results = Results

    def __init__(self, *args, **kwargs):

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
        self.solver_results = None

    def kernel(self, solver=None, init_guess=None, eris=None, construct_bath=True, chempot=None):
        """Run solver for a single BNO threshold.

        Parameters
        ----------
        solver : {'MP2', 'CISD', 'CCSD', 'CCSD(T)', 'FCI'}, optional
            Correlated solver.

        Returns
        -------
        results : DMETFragmentResults
        """
        solver = solver or self.base.solver
        if solver not in self.base.valid_solvers:
            raise ValueError("Unknown solver: %s" % solver)
        if self._dmet_bath is None or construct_bath:
            self.make_bath()

        cluster = self.make_cluster()
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

        self._results = results = self.Results(fid=self.id, wf=cluster_solver.wf, n_active=self.cluster.norb_active,
                dm1=cluster_solver.wf.make_rdm1(), dm2=cluster_solver.wf.make_rdm2())
        results.e1, results.e2 = self.get_dmet_energy_contrib()

        return results

    def get_solver_options(self, solver):
        solver_opts = {}
        solver_opts.update(self.opts.solver_options)
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

    def get_active_space_correlation_energy(self, eris=None):

        c_act = self.cluster.c_active
        if eris is None:
            with log_time(self.log.timing, "Time for AO->MO transformation: %s"):
                eris = self.base.get_eris_array(c_act)

        mo_core = self.cluster.c_frozen_occ
        mo_cas = self.cluster.c_active

        hcore = self.base.get_hcore()
        energy_core = self.base.mf.energy_nuc()
        if mo_core.size == 0:
            corevhf = 0
        else:
            core_dm = dot(mo_core, mo_core.conj().T) * 2
            corevhf = self.base.mf.get_veff(dm=core_dm)
            energy_core += einsum('ij,ji', core_dm, hcore).real
            energy_core += einsum('ij,ji', core_dm, corevhf).real * .5
        h1eff = dot(mo_cas.conj().T, hcore+corevhf, mo_cas)

        dm1 = self.results.dm1
        dm2 = self.results.dm2

        e1 = dot(dm1, h1eff).trace()
        e2 = einsum("pqrs,pqsr->", dm2, eris) * .5
        return energy_core + e1 + e2, energy_core

    def get_frag_hl_dm(self):
        c = dot(self.c_frag.T, self.mf.get_ovlp(), self.cluster.c_active)
        return dot(c, self.results.dm1, c.T)

    def get_nelectron_hl(self):
        return self.get_frag_hl_dm().trace()
