import dataclasses
import numpy as np

from vayesta.ewf.fragment import Fragment as EWF_Fragment
from dyson import MBLGF, MixedMBLGF, AufbauPrinciple, AuxiliaryShift, gf_moments_to_se_moments

@dataclasses.dataclass
class Options(EWF_Fragment.Options):
    pass

class Fragment(EWF_Fragment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def kernel(self, solver=None, init_guess=None):
        results = super().kernel(solver=solver, init_guess=init_guess)

        th, tp = results.moms
        if self.base.opts.sym_moms:
            th = 0.5 * (th + th.transpose(0,2,1))
            tp = 0.5 * (tp + tp.transpose(0,2,1))
            
        t0 = th[0]+tp[0]
        err = np.linalg.norm(t0 - np.eye(t0.shape[0]))
        
        print(t0- np.eye(t0.shape[0]))
        print("Error in zeroth GF moment: %s"%err)
        
        hermitian_mblgf = self.base.opts.hermitian_mblgf
        #self.base.log('Running block Lanczos to preserve GF moments')
        solverh = MBLGF(th, hermitian=hermitian_mblgf, log=self.base.log)
        solverp = MBLGF(tp, hermitian=hermitian_mblgf, log=self.base.log)
        solver = MixedMBLGF(solverh, solverp)
        solver.kernel()
        gf = solver.get_greens_function()   
        se = solver.get_self_energy()

        if self.base.opts.aux_shift_frag:
            #self.emb.log('Running auxiliary shift to preserve electron number')
            solver = AuxiliaryShift(th[1]+tp[1], se, self.nelectron)
            solver.kernel()
            gf = solver.get_greens_function()
            se = solver.get_self_energy()
            # Add logging? Calc dm and print elec number

        results.moms = th, tp
        results.self_energy = se
        results.greens_function = gf
        results.static_self_energy = th[1]+tp[1]
        if 0:
            nmom_se = self.base.opts.nmom_se    
            results.self_energy_moments = np.array([se.moment(i) for i in range(len(tp)-2)])
            results.static_self_energy = th[1]+tp[1]
        else:
            # gf_moms = th+tp
            # gf_moms[0] = np.eye(gf_moms[0].shape[0])
            results.static_self_energy, results.self_energy_moments = gf_moments_to_se_moments(th+tp)


        return results





