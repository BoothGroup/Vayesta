import unittest
import numpy as np
from pyscf.pbc import gto, scf, tools
from vayesta.misc import gdf
from vayesta.core import foldscf as k2bvk

#TODO: make_mo_coeff_real


class K2BVK_RHF_Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        L = 5.0
        n = 11
        cls.cell = gto.Cell()
        cls.cell.a = np.eye(3) * L
        cls.cell.mesh = [n, n, n]
        cls.cell.atom = 'He 3 2 3; He 1 1 1'
        cls.cell.basis = '6-31g'
        cls.cell.verbose = 0
        cls.cell.max_memory = 1e9
        cls.cell.build()
        cls.scell = tools.super_cell(cls.cell, [2,2,2])
        cls.kpts = cls.cell.make_kpts([2,2,2])
        cls.kmf = scf.KRHF(cls.cell, cls.kpts)
        cls.kmf.conv_tol = 1e-12
        cls.kmf.with_df = gdf.GDF(cls.cell, cls.kpts)
        cls.kmf.with_df.linear_dep_threshold = 1e-7
        cls.kmf.with_df.build()
        cls.kmf.kernel()
        cls.mf = k2bvk.fold_scf(cls.kmf)
        cls.smf = scf.RHF(cls.scell)
        cls.smf.conv_tol = 1e-12
        cls.smf = cls.smf.density_fit()
        cls.smf.kernel()

    @classmethod
    def teardownClass(cls):
        del cls.cell, cls.kpts, cls.mf, cls.kmf, cls.smf


    def _test_values(self, a, b, prec=8):
        self.assertAlmostEqual(np.abs(np.max(a-b)), 0.0, prec)

    def test_ovlp(self):
        self._test_values(self.mf.get_ovlp(), self.smf.get_ovlp())

    def test_hcore(self):
        self._test_values(self.mf.get_hcore(), self.smf.get_hcore())

    def test_veff(self):
        self._test_values(self.mf.get_veff(), self.smf.get_veff())

    def test_fock(self):
        scell, phase = k2bvk.get_phase(self.cell, self.kpts)
        dm = k2bvk.k2bvk_2d(self.kmf.get_init_guess(), phase)
        self._test_values(self.mf.get_fock(dm=dm), self.smf.get_fock(dm=dm))


class K2BVK_UHF_Tests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        L = 5.0
        n = 11
        cls.cell = gto.Cell()
        cls.cell.a = np.eye(3) * L
        cls.cell.mesh = [n, n, n]
        cls.cell.atom = 'He 3 2 3; He 1 1 1'
        cls.cell.basis = '6-31g'
        cls.cell.verbose = 0
        cls.cell.max_memory = 1e9
        cls.cell.build()
        cls.scell = tools.super_cell(cls.cell, [2,2,2])
        cls.kpts = cls.cell.make_kpts([2,2,2])
        cls.kmf = scf.KUHF(cls.cell, cls.kpts)
        cls.kmf.conv_tol = 1e-12
        cls.kmf.with_df = gdf.GDF(cls.cell, cls.kpts)
        cls.kmf.with_df.linear_dep_threshold = 1e-7
        cls.kmf.with_df.build()
        cls.kmf.kernel()
        cls.mf = k2bvk.fold_scf(cls.kmf)
        cls.smf = scf.UHF(cls.scell)
        cls.smf.conv_tol = 1e-12
        cls.smf = cls.smf.density_fit()
        cls.smf.kernel()

    @classmethod
    def teardownClass(cls):
        del cls.cell, cls.kpts, cls.mf, cls.kmf, cls.smf


    def _test_values(self, a, b, prec=8):
        if isinstance(a, (tuple, list)):
            for _a, _b in zip(a, b):
                self.assertAlmostEqual(np.abs(np.max(_a-_b)), 0.0, prec)
        else:
            self.assertAlmostEqual(np.abs(np.max(a-b)), 0.0, prec)

    def test_ovlp(self):
        self._test_values(self.mf.get_ovlp(), self.smf.get_ovlp())

    def test_hcore(self):
        self._test_values(self.mf.get_hcore(), self.smf.get_hcore())

    def test_veff(self):
        self._test_values(self.mf.get_veff(), self.smf.get_veff())

    def test_fock(self):
        scell, phase = k2bvk.get_phase(self.cell, self.kpts)
        dm = k2bvk.k2bvk_2d(self.kmf.get_init_guess(), phase)
        self._test_values(self.mf.get_fock(dm=dm), self.smf.get_fock(dm=dm))



if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
