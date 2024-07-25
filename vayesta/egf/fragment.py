import dataclasses
import numpy as np

from vayesta.ewf.fragment import Fragment as EWF_Fragment
from dyson import MBLGF, MixedMBLGF, MBLSE, MixedMBLSE, AufbauPrinciple, AuxiliaryShift, gf_moments_to_se_moments

@dataclasses.dataclass
class Options(EWF_Fragment.Options):
    pass

class Fragment(EWF_Fragment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def kernel(self, solver=None, init_guess=None):
        results = super().kernel(solver=solver, init_guess=init_guess)


        # if results.se_moments is not None and 0:
        #     se_static = results.se_static
        #     seh_mom, sep_mom = results.se_moments
        #     if self.base.opts.sym_moms:
        #         seh_mom = 0.5 * (seh_mom + seh_mom.transpose(0,2,1))
        #         sep_mom = 0.5 * (sep_mom + sep_mom.transpose(0,2,1))
        #     hermitian_mblgf = self.base.opts.hermitian_mblgf
        #     solverh = MBLSE(se_static, seh_mom, hermitian=hermitian_mblgf, log=self.base.log)
        #     solverp = MBLSE(se_static, sep_mom, hermitian=hermitian_mblgf, log=self.base.log)
        #     solver = MixedMBLGF(solverh, solverp)
        #     solver.kernel()
        #     gf = solver.get_greens_function()   
        #     se = solver.get_self_energy()

        #     # FIXME: Find clear and consistent way of storing cluster SE/GF moments (and possibly convert between them)
        #     results.static_self_energy = se_static
            
        # elif results.gf_moments is not None:
        #     th, tp = results.gf_moments
        #     if self.base.opts.sym_moms:
        #         th = 0.5 * (th + th.transpose(0,2,1))
        #         tp = 0.5 * (tp + tp.transpose(0,2,1))
                
        #     t0 = th[0]+tp[0]
        #     err = np.linalg.norm(t0 - np.eye(t0.shape[0]))
            
        #     print(t0- np.eye(t0.shape[0]))
        #     print("Error in zeroth GF moment: %s"%err)
            
        #     hermitian_mblgf = self.base.opts.hermitian_mblgf
        #     #self.base.log('Running block Lanczos to preserve GF moments')
        #     solverh = MBLGF(th, hermitian=hermitian_mblgf, log=self.base.log)
        #     solverp = MBLGF(tp, hermitian=hermitian_mblgf, log=self.base.log)
        #     solver = MixedMBLGF(solverh, solverp)
        #     solver.kernel()
        #     gf = solver.get_greens_function()   
        #     se = solver.get_self_energy()

        #     # FIXME: Find clear and consistent way of storing cluster SE/GF moments (and possibly convert between them)
        #     # Should they include the symmetrisation and shift?
        #     results.gf_moments = th, tp
        #     results.static_self_energy = th[1]+tp[1]

        # if self.base.opts.aux_shift_frag:
        #     #self.emb.log('Running auxiliary shift to preserve electron number')
        #     solver = AuxiliaryShift(th[1]+tp[1], se, self.nelectron)
        #     solver.kernel()
        #     gf = solver.get_greens_function()
        #     se = solver.get_self_energy()
        #     # Add logging? Calc dm and print elec number

        
        # results.self_energy = se
        # results.greens_function = gf
        
        # if 1:
        #     nmom_se = self.base.opts.nmom_se    
        #     results.se_moments = np.array([se.moment(i) for i in range(len(tp)-2)])
        #     results.static_self_energy = th[1]+tp[1]
        # else:
        #     gf_moms = th+tp
        #     gf_moms[0] = np.eye(gf_moms[0].shape[0])
    

        # #results.static_self_energy, results.se_moments = gf_moments_to_se_moments(th+tp)
        # #results.static_self_energy = 

        return results





