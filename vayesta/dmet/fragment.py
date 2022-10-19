import dataclasses
from timeit import default_timer as timer

# External libaries
import numpy as np

from vayesta.core.qemb import Fragment
from vayesta.core.bath import BNO_Threshold

from vayesta.core import ao2mo
from vayesta.core.util import *


# We might want to move the useful things from here into core, since they seem pretty general.

class DMETFragmentExit(Exception):
    pass



class DMETFragment(Fragment):
    @dataclasses.dataclass
    class Results(Fragment.Results):
        n_active: int = None
        e1: float = None
        e2: float = None
        dm1: np.ndarray = None
        dm2: np.ndarray = None

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

    def kernel(self, solver=None, init_guess=None, eris=None, seris_ov=None, construct_bath=True, chempot=None):
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
        #get_solver_class(self.mf, solver)
        if self._dmet_bath is None or construct_bath:
            self.make_bath()

        cluster = self.make_cluster()
        # If we want to reuse previous info for initial guess and eris we'd do that here...
        # We can now overwrite the orbitals from last BNO run:
        self._c_active_occ = cluster.c_active_occ
        self._c_active_vir = cluster.c_active_vir

        if solver is None:
            return None

        cluster_solver = self.get_solver(solver)
        # Chemical potential
        if chempot is not None:
            px = self.get_fragment_projector(self.cluster.c_active)
            if isinstance(px, tuple):
                cluster_solver.v_ext = (-chempot * px[0], -chempot * px[1])
            else:
                cluster_solver.v_ext = -chempot * px

        with log_time(self.log.info, ("Time for %s solver:" % solver) + " %s"):
            if self.opts.screening:
                cluster_solver.kernel()
            else:
                cluster_solver.kernel()

        self._results = results = self.Results(fid=self.id, wf=cluster_solver.wf, n_active=self.cluster.norb_active,
                dm1=cluster_solver.wf.make_rdm1(), dm2=cluster_solver.wf.make_rdm2())
        if eris is None:
            # We can cache these if they're used in the actualy calculation.
            eris = cluster_solver.hamil.get_eris_bare()
        results.e1, results.e2 = self.get_dmet_energy_contrib(eris=eris)

        return results

    def get_solver_options(self, solver):
        solver_opts = {}
        solver_opts.update(self.opts.solver_options)
        return solver_opts

    def get_dmet_energy_contrib(self, eris=None):
        """Calculate the contribution of this fragment to the overall DMET energy.

        TODO: use core.qemb.fragment.get_fragment_dmet_energy instead?
        """
        # Projector to the impurity in the active basis.
        P_imp = self.get_fragment_projector(self.cluster.c_active)
        c_act = self.cluster.c_active
        if eris is None:
            eris = self._eris
        if eris is None:
            with log_time(self.log.timing, "Time for AO->MO transformation: %s"):
                eris = self.base.get_eris_array(c_act)
        if not isinstance(eris, np.ndarray):
            self.log.debugv("Extracting ERI array from CCSD ERIs object.")
            eris = ao2mo.helper.get_full_array(eris, c_act)

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
