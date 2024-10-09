import unittest

import numpy as np
from vayesta import egf
from vayesta.core.util import AbstractMethodError, cache
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems
try:
    from dyson import CCSD, FCI, MBLGF, AuxiliaryShift, AufbauPrinciple, NullLogger
except ImportError:
    pytest.skip("Requires dyson")

class Test_Full_Bath_FCI(TestCase):
    NFRAG = 2
    NMOM_MAX_GF = 4
    NMOM_MAX_SE = 4
    ETA = 1e-2
    system = testsystems.hubb_10_u2
    solver = 'FCI'
    EXPR = FCI
    
    @classmethod
    def setUpClass(cls):

            
        cls.mf = cls.system.rhf()

    # def __init__(self):
    #     raise AbstractMethodError()
    
    @classmethod
    @cache
    def exact(cls, mf):
        try:
            from dyson import CCSD, FCI, MBLGF, MixedMBLGF, AuxiliaryShift, NullLogger
        except ImportError:
            pytest.skip("Requires dyson")

        expr = cls.EXPR["1h"](mf)
        th = expr.build_gf_moments(cls.NMOM_MAX_GF) 
        expr = cls.EXPR["1p"](mf)
        tp = expr.build_gf_moments(cls.NMOM_MAX_GF)

        solverh = MBLGF(th, log=NullLogger())
        solverp = MBLGF(tp, log=NullLogger())
        solver = MixedMBLGF(solverh, solverp)
        solver.kernel()
        se = solver.get_self_energy()
        Shift = AufbauPrinciple if cls.solver == 'CCSD' else AuxiliaryShift
        solver = Shift(th[1]+tp[1], se, mf.mol.nelectron, log=NullLogger())
        solver.kernel()
        gf = solver.get_greens_function()
        se = solver.get_self_energy()
        return se, gf
    
    @classmethod
    @cache
    def emb(cls, mf):
        opts = dict(solver=cls.solver, proj=1, aux_shift=True, use_sym=True)
        bath_opts = dict(bathtype='full', dmet_threshold=1e-12)
        solver_opts =dict(conv_tol=1e-15, n_moments=(cls.NMOM_MAX_GF, cls.NMOM_MAX_GF))
        emb = egf.EGF(mf, **opts, bath_options=bath_opts, solver_options=solver_opts)
        nimage = [emb.mf.mol.natm//cls.NFRAG, 1, 1]
        emb.symmetry.set_translations(nimage)
        with emb.site_fragmentation() as f:
            f.add_atomic_fragment(list(range(cls.NFRAG)))
        emb.kernel()
        return emb
    
    def test_ea_ip(self):
        se_exact, gf_exact = self.exact(self.mf)
        emb = self.emb(self.mf)
        se_emb, gf_emb = emb.self_energy, emb.gf

        # Test physical QP energies
        ip_exact = gf_exact.physical().occupied().energies[-1]
        ea_exact = gf_exact.physical().virtual().energies[0]
        ip_emb = gf_emb.physical().occupied().energies[-1]
        ea_emb = gf_emb.physical().virtual().energies[0]
        self.assertAlmostEqual(ip_exact, ip_emb, places=5)
        self.assertAlmostEqual(ea_exact, ea_emb, places=5)

    def test_perturbed_mo_energy(self):
        se_exact, gf_exact = self.exact(self.mf)
        emb = self.emb(self.mf)
        se_emb, gf_emb = emb.self_energy, emb.gf

        gf_energies_exact = gf_exact.as_perturbed_mo_energy()
        gf_energies_emb = gf_emb.as_perturbed_mo_energy()

        self.assertAllclose(gf_energies_exact, gf_energies_emb, atol=1e-5)


    def test_gf_moments(self):
        se_exact, gf_exact = self.exact(self.mf)
        emb = self.emb(self.mf)
        se_emb, gf_emb = emb.self_energy, emb.gf

        th_exact = np.array([gf_exact.occupied().moment(i) for i in range(self.NMOM_MAX_GF)])
        tp_exact = np.array([gf_exact.virtual().moment(i) for i in range(self.NMOM_MAX_GF)])
        th_emb = np.array([gf_emb.occupied().moment(i) for i in range(self.NMOM_MAX_GF)])
        tp_emb = np.array([gf_emb.virtual().moment(i) for i in range(self.NMOM_MAX_GF)])
        print(np.linalg.norm(th_exact+tp_exact - th_emb-tp_emb, axis=(1,2)))

        #self.assertAllclose(th_exact, th_emb, atol=1e-5)
        #self.assertAllclose(tp_exact, tp_emb, atol=1e-5)
        self.assertAllclose(th_exact+tp_exact, th_emb+tp_emb, atol=1e-4)

    def test_se_moments(self):
        se_exact, gf_exact = self.exact(self.mf)
        emb = self.emb(self.mf)
        se_emb, gf_emb = emb.self_energy, emb.gf

        se_mom_exact = [se_exact.moment(i) for i in range(self.NMOM_MAX_SE)]
        se_mom_emb = [se_emb.moment(i) for i in range(self.NMOM_MAX_SE)]
        self.assertAllclose(se_mom_exact, se_mom_emb, atol=1e-4)

    def test_static_potential(self):
        se_exact, gf_exact = self.exact(self.mf)
        emb = self.emb(self.mf)
        se_emb, gf_emb = emb.self_energy, emb.gf

        static_potential_exact = se_exact.as_static_potential(self.mf.mo_energy, eta=self.ETA)
        static_potential_emb = se_emb.as_static_potential(self.mf.mo_energy, eta=self.ETA)
        self.assertAllclose(static_potential_exact, static_potential_emb, atol=1e-5)




class Test_Full_Bath_CCSD(Test_Full_Bath_FCI):


    system = testsystems.hubb_10_u2
    solver = 'CCSD'
    EXPR = CCSD
    

if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()



# class Test_Full_Bath_CCSD2(EWDMET_Test):

#     def __init__(self, args, kwargs):
#         self.mf = testsystems.water_sto3g.rhf()
#         self.solver = 'CCSD'
#         self.expr = CCSD
