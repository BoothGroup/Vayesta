''' Tests for a helium cell using vayesta/misc/gdf.py, copied from PySCF
'''

# --- Standard library
import unittest

# --- NumPy
import numpy as np

# --- PySCF
from pyscf.pbc import gto, scf, df

# --- Vayesta
from vayesta.misc import gdf


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.a = np.eye(3) * 5.0
        cell.mesh = np.array([11, 11, 11])
        cell.atom = 'He 3 2 3; He 1 1 1'
        cell.basis = 'cc-pvdz'
        cell.verbose = 0
        cell.max_memory = 1000
        cell.build(0, 0)

        mf0 = scf.RHF(cell)
        mf0.exxdiv = 'vcut_sph'

        np.random.seed(1)
        kpts = np.random.random((5, 3))
        kpts[0] = 0
        kpts[3] = kpts[0] - kpts[1] + kpts[2]
        kpts[4] *= 1e-5

        kmdf_ref = df.GDF(cell)
        kmdf_ref.linear_dep_threshold = 1e-7
        kmdf_ref.auxbasis = 'weigend'
        kmdf_ref.kpts = kpts

        kmdf = gdf.GDF(cell)
        kmdf.linear_dep_threshold = 1e-7
        kmdf.auxbasis = 'weigend'
        kmdf.kpts = kpts

        cls.cell = cell
        cls.kmdf = kmdf
        cls.kmdf_ref = kmdf_ref
        cls.kpts = kpts


    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kmdf, cls.kmdf_ref, cls.kpts


    def test_get_eri_gamma(self):
        cell, kmdf, kmdf_ref, kpts = self.cell, self.kmdf, self.kmdf_ref, self.kpts

        odf = gdf.GDF(cell)
        odf.linear_dep_threshold = 1e-7
        odf.auxbasis = 'weigend'
        eri0000 = odf.get_eri()
        eri1111 = kmdf.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        eri4444 = kmdf.get_eri((kpts[4],kpts[4],kpts[4],kpts[4]))

        odf_ref = df.GDF(cell)
        odf_ref.linear_dep_threshold = 1e-7
        odf_ref.auxbasis = 'weigend'
        eri0000_ref = odf_ref.get_eri()
        eri1111_ref = kmdf_ref.get_eri((kpts[0],kpts[0],kpts[0],kpts[0]))
        eri4444_ref = kmdf_ref.get_eri((kpts[4],kpts[4],kpts[4],kpts[4]))

        self.assertTrue(np.allclose(eri0000, eri0000_ref, atol=1e-7))
        self.assertTrue(np.allclose(eri1111, eri1111_ref, atol=1e-7))
        self.assertTrue(np.allclose(eri4444, eri4444_ref, atol=1e-7))

    def test_get_eri_1111(self):
        cell, kmdf, kmdf_ref, kpts = self.cell, self.kmdf, self.kmdf_ref, self.kpts
        eri1111 = kmdf.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        eri1111_ref = kmdf_ref.get_eri((kpts[1],kpts[1],kpts[1],kpts[1]))
        check2 = kmdf.get_eri((kpts[1]+5e-8,kpts[1]+5e-8,kpts[1],kpts[1]))
        check2_ref = kmdf_ref.get_eri((kpts[1]+5e-8,kpts[1]+5e-8,kpts[1],kpts[1]))
        self.assertTrue(np.allclose(eri1111, check2, atol=1e-7))
        self.assertTrue(np.allclose(eri1111_ref, check2_ref, atol=1e-7))
        self.assertTrue(np.allclose(eri1111, eri1111_ref, atol=1e-7))
        self.assertTrue(np.allclose(check2, check2_ref, atol=1e-7))

    def test_get_eri_0011(self):
        cell, kmdf, kmdf_ref, kpts = self.cell, self.kmdf, self.kmdf_ref, self.kpts
        eri0011 = kmdf.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        eri0011_ref = kmdf_ref.get_eri((kpts[0],kpts[0],kpts[1],kpts[1]))
        self.assertTrue(np.allclose(eri0011, eri0011_ref, atol=1e-7))

    def test_get_eri_0110(self):
        cell, kmdf, kmdf_ref, kpts = self.cell, self.kmdf, self.kmdf_ref, self.kpts
        eri0110 = kmdf.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        eri0110_ref = kmdf_ref.get_eri((kpts[0],kpts[1],kpts[1],kpts[0]))
        check2 = kmdf.get_eri((kpts[0]+5e-8,kpts[1]+5e-8,kpts[1],kpts[0]))
        check2_ref = kmdf_ref.get_eri((kpts[0]+5e-8,kpts[1]+5e-8,kpts[1],kpts[0]))
        self.assertTrue(np.allclose(eri0110, check2, atol=1e-7))
        self.assertTrue(np.allclose(eri0110_ref, check2_ref, atol=1e-7))
        self.assertTrue(np.allclose(eri0110, eri0110_ref, atol=1e-7))
        self.assertTrue(np.allclose(check2, check2_ref, atol=1e-7))

    def test_get_eri_0123(self):
        cell, kmdf, kmdf_ref, kpts = self.cell, self.kmdf, self.kmdf_ref, self.kpts
        eri0123 = kmdf.get_eri(kpts[:4])
        eri0123_ref = kmdf.get_eri(kpts[:4])
        self.assertTrue(np.allclose(eri0123, eri0123_ref, atol=1e-7))



if __name__ == '__main__':
    print('Unit tests for GDF')
    unittest.main()
