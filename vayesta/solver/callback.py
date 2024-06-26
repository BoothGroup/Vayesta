import dataclasses
from typing import Callable
import numpy as np

from vayesta.core.types import CISD_WaveFunction, CCSD_WaveFunction, FCI_WaveFunction, RDM_WaveFunction
from vayesta.solver.solver import ClusterSolver

class CallbackSolver(ClusterSolver):
    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        # Need to specify a type for this to work
        callback: int = None

    def kernel(self, *args, **kwargs):
        mf_clus, frozen = self.hamil.to_pyscf_mf(allow_dummy_orbs=True, allow_df=True)
        results = self.opts.callback(mf_clus)

        # Build appropriate wavefunction object
        if 'civec' in results:
            self.log.info("FCI WaveFunction found in callback results.")
            wf = FCI_WaveFunction(self.hamil.mo, results['civec'])
        elif 't1' in results and 't2' in results:
            self.log.info("CCSD WaveFunction found in callback results.")
            t1, t2 = results['t1'], results['t2']
            if 'l1' in results and 'l2' in results:
                l1, l2 = results['l1'], results['l2']
            else:
                l1, l2 = None, None
            wf = CCSD_WaveFunction(self.hamil.mo, t1, t2, l1=l1, l2=l2)
        elif 'c0' in results and 'c1' in results and 'c2' in results:
            self.log.info("CISD WaveFunction found in callback results.")
            c0, c1, c2 = results['c0'], results['c1'], results['c2']
            wf = CISD_WaveFunction(self.hamil.mo, c0, c1, c2)
        elif 'dm1' in results and 'dm2' in results:
            self.log.info("RDM WaveFunction found in callback results.")
            dm1, dm2 = results['dm1'], results['dm2']
            wf = RDM_WaveFunction(self.hamil.mo, dm1, dm2)
        else:
            self.log.warn("No wavefunction results returned by callback!")

        if 'gf_hole_moments' in results:
            self.log.info("Green's function hole moments found in callback results.")
            self.gf_hole_moments = results['gf_hole_moments']
        if 'gf_particle_moments' in results:
            self.log.info("Green's function particle moments found in callback results.")
            self.gf_particle_moments = results['gf_particle_moments']
        if 'se_hole_moments' in results:
            self.log.info("Self-energy hole moments found in callback results.")
            self.hole_moments = results['se_hole_moments']
        if 'se_particle_moments' in results:
            self.log.info("Self-energy particle moments found in callback results.")
            self.se_particle_moments = results['se_particle_moments']
            
        results['wf'] = wf
        self.wf = wf
        self.converged = results['converged'] if 'converged' in results else False
        self.callback_results = results
