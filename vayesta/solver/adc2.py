import dataclasses

import numpy as np
try:
    import dyson
    from dyson import MBLSE, ADC2
    dyson.quiet()
except ImportError:
    print("Dyson not found - required for ADC solver")


from vayesta.core.types import RRDM_WaveFunction, SE_MomentRep
from vayesta.core.util import log_time, brange, einsum
from vayesta.solver.solver import ClusterSolver, UClusterSolver


class RADC2_Solver(ClusterSolver):

    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        nmom: (int, int) = (11,11)

    def kernel(self):
        mf_clus, frozen = self.hamil.to_pyscf_mf(allow_dummy_orbs=True, allow_df=True)
        exprh = ADC2.hole.from_mf(mf_clus)
        exprp = ADC2.particle.from_mf(mf_clus)

        se_statics, overlaps, se_moms = [], [], []
        self.log.info("Calculating cluster ADC(2) self-energy moments")

        heff, eris = self.hamil.get_integrals(with_vext=True)
        
        with log_time(self.log.timing, "Time for self-energy hole moments: %s"):
            exprh = ADC2.hole.from_mf(mf_clus)
            ovlph, seh_static = exprh.build_gf_moments(2)
            ooov = self.hamil._get_eris_block(eris, 'ooov')
            seh_moms = exprh.build_se_moments(self.opts.nmom[0], ooov=ooov)
            se_statics.append(seh_static)
            overlaps.append(ovlph)
            se_moms.append(seh_moms)

        with log_time(self.log.timing, "Time for self-energy particle moments: %s"):
            exprp = ADC2.particle.from_mf(mf_clus)
            ovlpp, sep_static = exprp.build_gf_moments(2)
            vvvo = self.hamil._get_eris_block(eris, 'vvvo')
            sep_moms = exprp.build_se_moments(self.opts.nmom[1], vvvo=vvvo)
            se_statics.append(sep_static)
            overlaps.append(ovlpp)
            se_moms.append(sep_moms) 

        nmo = se_statics[0].shape[-1]
        dm1 = np.zeros((nmo, nmo))
        dm2 = np.zeros((nmo, nmo, nmo, nmo))
        self.wf = RRDM_WaveFunction(self.hamil.mo, dm1, dm2)

        self.se = SE_MomentRep(se_statics, se_moms, overlap=overlaps)

        results = dict(wf=self.wf, se=self.se)
        self.converged = True
        return results
