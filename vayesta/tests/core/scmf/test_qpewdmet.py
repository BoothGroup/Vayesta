import unittest
import pytest
import numpy as np
import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


class Test_FCI_H6_1Projector(TestCase):
    nmom_fci = (4, 4)
    proj = 1
    nfrag = 1
    @classmethod
    def setUpClass(cls):        
        cls.mf = testsystems.h6_sto6g.rhf()
        cls.run_dyson()

    @classmethod
    def run_dyson(cls):
        try:
            from dyson import FCI, MBLGF, MixedMBLGF, NullLogger, AuxiliaryShift
        except ImportError:
            pytest.skip("Requires dyson")
        cls.fci_ip = FCI["1h"](cls.mf)
        cls.fci_ip_moms = cls.fci_ip.build_gf_moments(cls.nmom_fci[0])

        cls.fci_ea = FCI["1p"](cls.mf)
        cls.fci_ea_moms = cls.fci_ea.build_gf_moments(cls.nmom_fci[1])
        solverh = MBLGF(cls.fci_ip_moms, log=NullLogger())
        solverp = MBLGF(cls.fci_ea_moms, log=NullLogger())
        solver = MixedMBLGF(solverh, solverp)
        solver.kernel()
        se = solver.get_self_energy()
        solver = AuxiliaryShift(cls.fci_ip_moms[1]+cls.fci_ea_moms[1], se, cls.mf.mol.nelectron)
        solver.kernel()
        cls.gf = solver.get_greens_function()
        cls.se = solver.get_self_energy()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        del cls.fci_ip
        del cls.fci_ip_moms
        del cls.fci_ea
        del cls.fci_ea_moms
        cls.emb.cache_clear()


    @classmethod
    @cache
    def emb(cls):
        emb = vayesta.ewf.EWF(cls.mf, bath_options=dict(bathtype='full'), solver_options=dict(n_moments=cls.nmom_fci), solver="FCI")
        emb.qpewdmet_scmf(proj=cls.proj, maxiter=1)
        with emb.iaopao_fragmentation() as f:
            for i in range(0,cls.mf.mol.natm,cls.nfrag):
                f.add_atomic_fragment(list(range(i,i+cls.nfrag)))
        emb.kernel()
        return emb
    
    #def test_energy(self):
        #emb = self.emb()
        #self.assertAllclose(emb.e_tot, self.fci_ip.e_ci, atol=1e-5, rtol=1e-4)

    def test_self_energy(self):
        emb = self.emb()
        n_se_mom = 4
        se_moms = np.array([self.se.moment(i) for i in range(n_se_mom)])
        emb_se_moms = np.array([emb.with_scmf.se_shifted.moment(i) for i in range(n_se_mom)])
        self.assertTrue(np.allclose(se_moms, emb_se_moms, atol=1e-4))
        
    def test_static_self_energy(self):
        emb = self.emb()
        static_self_energy = self.fci_ip_moms[1] + self.fci_ea_moms[1] 
        emb_static_self_energy = self.mf.mo_coeff.T @ self.mf.get_fock() @ self.mf.mo_coeff + emb.with_scmf.static_self_energy 
        self.assertTrue(np.allclose(static_self_energy, emb_static_self_energy, atol=1e-4))
    
    def test_static_potential(self):
        emb = self.emb()
        static_potential = self.se.as_static_potential(self.mf.mo_energy, eta=1e-2)
        emb_static_potential = emb.with_scmf.se_shifted.as_static_potential(self.mf.mo_energy, eta=1e-2)
        self.assertTrue(np.allclose(static_potential, emb_static_potential, atol=1e-4))

    def test_greens_function(self):
        emb = self.emb()
        gf = emb.with_scmf.gf
        gf_moms = np.array([gf.moment(i) for i in range(self.nmom_fci[0])])
        self.assertTrue(np.allclose(gf_moms, self.fci_ip_moms+self.fci_ea_moms, atol=1e-4))

    def test_gap(self):
        emb = self.emb()
        gap = self.gf.physical().virtual().energies[0] - self.gf.physical().occupied().energies[-1]
        emb_gap = emb.with_scmf.gf.physical().virtual().energies[0] - emb.with_scmf.gf.physical().occupied().energies[-1]
        self.assertTrue(np.allclose(gap, emb_gap, atol=1e-4))


class Test_FCI_H6_2Projector(Test_FCI_H6_1Projector):
    proj = 2
    nfrag = 6

class Test_FCI_Hubbard10_1Projector(Test_FCI_H6_1Projector):
    @classmethod
    def setUpClass(cls):        
        cls.mf = testsystems.hubb_10_u2.rhf()
        cls.run_dyson()

    @classmethod
    @cache
    def emb(cls):
        emb = vayesta.ewf.EWF(cls.mf, bath_options=dict(bathtype='full'), solver_options=dict(n_moments=cls.nmom_fci), solver="FCI")
        emb.qpewdmet_scmf(proj=cls.proj, use_sym=True, maxiter=1)
        nimages = [cls.mf.mol.natm//cls.nfrag, 1, 1]
        emb.symmetry.set_translations(nimages)
        with emb.site_fragmentation() as f:
            f.add_atomic_fragment(list(range(cls.nfrag)))
        emb.kernel()
        return emb

class Test_FCI_Hubbard10_2Projector(Test_FCI_Hubbard10_1Projector):
    proj = 2
    nfrag = 10

if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
