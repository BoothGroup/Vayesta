import dataclasses

import numpy as np
try:
    import momentGW
    from momentGW import GW
    momentGW.logging.silent = 1
    import dyson
    from dyson import MBLSE
    dyson.quiet()
except ImportError:
    print("Dyson and momentGW not found - required for GW solver")


from vayesta.core.types import RRDM_WaveFunction, SE_MomentRep
from vayesta.core.util import log_time, brange, einsum
from vayesta.solver.solver import ClusterSolver, UClusterSolver

class RGW_Solver(ClusterSolver):

    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        polarizability: str = 'drpa'
        optimize_chempot: bool = False
        fock_loop: bool = False

        n_moments: (int, int) = (11,11)



    def kernel(self):
        mf_clus, frozen = self.hamil.to_pyscf_mf(allow_dummy_orbs=True, allow_df=True)
        gw = GW(mf_clus)
        gw.polarizability = self.opts.polarizability


        self.log.info("Tranforming integrals to cluster MO basis")
        integrals = gw.ao2mo()
        self.log.info("Calculating cluster GW self-energy moments")
        se_static = gw.build_se_static(integrals)
        with log_time(self.log.timing, "Time for self-energy moments: %s"):
            se_moms = gw.build_se_moments(self.opts.n_moments[0], integrals)

        nmo = se_static.shape[-1]
        dm1 = np.zeros((nmo, nmo))
        dm2 = np.zeros((nmo, nmo, nmo, nmo))
        self.wf = RRDM_WaveFunction(self.hamil.mo, dm1, dm2)

        self.se = SE_MomentRep([se_static, se_static], se_moms)

        results = dict(wf=self.wf, se=self.se)
        self.converged = True
        return results

        
        


    

        

