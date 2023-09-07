import dataclasses

import numpy as np
import scipy.linalg

from vayesta.core.spinalg import hstack_matrices
from vayesta.core.types import Orbitals, Cluster
from vayesta.core.util import dot, einsum
from vayesta.solver.ccsd import RCCSD_Solver
from vayesta.solver.fci import FCI_Solver


class TRCCSD_Solver(RCCSD_Solver):
    @dataclasses.dataclass
    class Options(RCCSD_Solver.Options):
        fci_opts: dict = dataclasses.field(default_factory=dict)
        c_cas_occ: np.array = None
        c_cas_vir: np.array = None

    def get_callback(self):
        # Just need to generate FCI soln to reduced cluster, then make callback function.
        orbs = Orbitals(hstack_matrices(self.opts.c_cas_occ, self.opts.c_cas_vir), occ=self.opts.c_cas_occ.shape[-1])
        # Dummy cluster without any environmental orbitals.
        tclus = Cluster(orbs, None)
        tham = self.hamil.with_new_cluster(tclus)
        fci = FCI_Solver(tham)
        # Need to set v_ext for the FCI calculation. Just want rotation into our current basis.
        ro = dot(self.hamil.cluster.c_active_occ.T, self.hamil.orig_mf.get_ovlp(), tclus.c_active_occ)
        rv = dot(self.hamil.cluster.c_active_vir.T, self.hamil.orig_mf.get_ovlp(), tclus.c_active_vir)
        r = scipy.linalg.block_diag(ro, rv)

        if self.v_ext is not None:
            fci.v_ext = dot(r.T, self.v_ext, r)

        fci.kernel()
        if not fci.converged:
            self.log.error("FCI not converged!")
        wf = fci.wf.as_ccsd()
        # Now have FCI solution in the fragment, just need to write tailor function.
        # Delete everything for the FCI.
        del fci, tham

        def tailor_func(kwargs):
            cc = kwargs["mycc"]
            t1, t2 = kwargs["t1new"], kwargs["t2new"]
            # Rotate & project CC amplitudes to CAS
            t1_cc = einsum("IA,Ii,Aa->ia", t1, ro, rv)
            t2_cc = einsum("IJAB,Ii,Jj,Aa,Bb->ijab", t2, ro, ro, rv, rv)
            # Take difference wrt to FCI
            dt1 = wf.t1 - t1_cc
            dt2 = wf.t2 - t2_cc
            # Rotate back to CC space
            dt1 = einsum("ia,Ii,Aa->IA", dt1, ro, rv)
            dt2 = einsum("ijab,Ii,Jj,Aa,Bb->IJAB", dt2, ro, ro, rv, rv)
            # Add correction
            t1 += dt1
            t2 += dt2
            cc._norm_dt1 = np.linalg.norm(dt1)
            cc._norm_dt2 = np.linalg.norm(dt2)

        return tailor_func

    def print_extra_info(self, mycc):
        self.log.debug("Tailored CC: |dT1|= %.2e |dT2|= %.2e", mycc._norm_dt1, mycc._norm_dt2)
        del mycc._norm_dt1, mycc._norm_dt2
