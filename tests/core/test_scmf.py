import unittest
import numpy as np
from pyscf import gto, scf
from vayesta import ewf


class SCMFTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = 'N 0 0 0; N 0 0 1.1'
        cls.mol.basis = '6-31g'
        cls.mol.verbose = 0
        cls.mol.max_memory = 1e9
        cls.mol.build()
        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def test_pdmet(self):
        emb = ewf.EWF(self.mf, make_rdm1=True, bath_type=None)
        emb.sao_fragmentation()
        emb.make_all_atom_fragments()
        emb.pdmet_scmf(etol=1e-10, dtol=1e-8)
        emb.kernel()

        self.assertTrue(emb.with_scmf.converged)
        self.assertAlmostEqual(emb.with_scmf.e_tot_oneshot, -109.06727638760327, 8)
        self.assertAlmostEqual(emb.with_scmf.e_tot, -109.06845425753974, 8)

    def test_brueckner(self):
        emb = ewf.EWF(self.mf, make_rdm1=True, bath_type=None)
        emb.sao_fragmentation()
        emb.make_all_atom_fragments()
        emb.brueckner_scmf(etol=1e-10, dtol=1e-8)
        emb.kernel()

        self.assertTrue(emb.with_scmf.converged)
        self.assertAlmostEqual(emb.with_scmf.e_tot_oneshot, -109.06727638760327, 8)
        self.assertAlmostEqual(emb.with_scmf.e_tot, -109.06844454464843, 8)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
