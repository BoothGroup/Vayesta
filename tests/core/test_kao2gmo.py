import unittest
import numpy as np
from pyscf import mp, cc
from pyscf.pbc import gto, scf, tools
from vayesta.core.ao2mo import kao2gmo
from vayesta.misc import gdf

#TODO: make sure that the example includes sufficient imaginary parts?
#TODO: 2d systems with negative part
#TODO: python driver


class KAO2GMOTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cell = gto.Cell()
        cls.cell.atom = 'He 0 0 0'
        cls.cell.basis = '6-31g'
        cls.cell.a = np.eye(3) * 3
        cls.cell.verbose = 0
        cls.cell.build()
        cls.kpts = cls.cell.make_kpts([2,2,2])
        cls.mf = scf.KRHF(cls.cell, cls.kpts)
        cls.mf.with_df = gdf.GDF(cls.cell, cls.kpts)
        cls.mf.with_df.build()
        cls.mf.kernel()
        cls.gmf = tools.k2gamma.k2gamma(cls.mf)
        cls.gmf = cls.gmf.density_fit()

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kpts, cls.mf

    def test_gdf_to_eris(self):
        mf = self.gmf
        mo_coeff_occ = mf.mo_coeff[:, mf.mo_occ > 0]
        mo_coeff_vir = mf.mo_coeff[:, mf.mo_occ == 0]
        eri0 = kao2gmo.gdf_to_eris(self.mf.with_df, mf.mo_coeff, nocc=mo_coeff_occ.shape[1], only_ovov=False)
        for key in eri0.keys():
            coeffs = tuple([mo_coeff_occ, mo_coeff_vir][k == 'v'] for k in key)
            eri1 = mf.with_df.ao2mo(coeffs, compact=False).reshape([c.shape[1] for c in coeffs])
            self.assertAlmostEqual(np.max(np.abs(eri0[key]-eri1)), 0.0, 10)

    def test_mp2_eris(self):
        mf = self.gmf
        cm = mp.mp2.MP2(mf)
        mo_coeff_occ = mf.mo_coeff[:, mf.mo_occ > 0]
        mo_coeff_vir = mf.mo_coeff[:, mf.mo_occ == 0]
        eri0 = kao2gmo.gdf_to_pyscf_eris(mf, self.mf.with_df, cm)
        for key in ['ovov']:
            coeffs = tuple([mo_coeff_occ, mo_coeff_vir][k == 'v'] for k in key)
            eri1 = mf.with_df.ao2mo(coeffs, compact=False).reshape([c.shape[1] for c in coeffs])
            self.assertAlmostEqual(np.max(np.abs(getattr(eri0, key)-eri1)), 0.0, 10)

    def test_rccsd_eris(self):
        mf = self.gmf
        cm = cc.rccsd.RCCSD(mf)
        mo_coeff_occ = mf.mo_coeff[:, mf.mo_occ > 0]
        mo_coeff_vir = mf.mo_coeff[:, mf.mo_occ == 0]
        eri0 = kao2gmo.gdf_to_pyscf_eris(mf, self.mf.with_df, cm)
        for key in ['ovov', 'oovv', 'ovvo', 'ovoo', 'oooo', 'ovvv', 'vvvv']:
            coeffs = tuple([mo_coeff_occ, mo_coeff_vir][k == 'v'] for k in key)
            eri1 = mf.with_df.ao2mo(coeffs, compact=False).reshape([c.shape[1] for c in coeffs])
            self.assertAlmostEqual(np.max(np.abs(getattr(eri0, key)-eri1)), 0.0, 10)

    def test_dfccsd_eris(self):
        mf = self.gmf
        cm = cc.dfccsd.RCCSD(mf)
        mo_coeff_occ = mf.mo_coeff[:, mf.mo_occ > 0]
        mo_coeff_vir = mf.mo_coeff[:, mf.mo_occ == 0]
        eri0 = kao2gmo.gdf_to_pyscf_eris(mf, self.mf.with_df, cm)
        for key in ['ovov', 'oovv', 'ovvo', 'ovoo', 'oooo']:
            coeffs = tuple([mo_coeff_occ, mo_coeff_vir][k == 'v'] for k in key)
            eri1 = mf.with_df.ao2mo(coeffs, compact=False).reshape([c.shape[1] for c in coeffs])
            self.assertAlmostEqual(np.max(np.abs(getattr(eri0, key)-eri1)), 0.0, 10)
        # ovvv is compressed with sym=True:
        coeffs = [mo_coeff_occ, mo_coeff_vir, mo_coeff_vir, mo_coeff_vir]
        eri1 = mf.with_df.ao2mo(coeffs, compact=True)
        eri1 = eri1.reshape([coeffs[0].shape[1], coeffs[1].shape[1], -1])
        self.assertAlmostEqual(np.max(np.abs(eri0.ovvv-eri1)), 0.0, 10)
        # (vv|L)(L|vv) -> (vv|vv) contraction is performed late:
        coeffs = [mo_coeff_vir, mo_coeff_vir, mo_coeff_vir, mo_coeff_vir]
        eri1 = mf.with_df.ao2mo(coeffs, compact=True)
        eri0_vvvv = np.dot(eri0.vvL, eri0.vvL.T)
        self.assertAlmostEqual(np.max(np.abs(eri0_vvvv-eri1)), 0.0, 10)

    def test_ccsd_eris(self):
        mf = self.gmf
        cm = cc.ccsd.CCSD(mf)
        mo_coeff_occ = mf.mo_coeff[:, mf.mo_occ > 0]
        mo_coeff_vir = mf.mo_coeff[:, mf.mo_occ == 0]
        eri0 = kao2gmo.gdf_to_pyscf_eris(mf, self.mf.with_df, cm)
        for key in ['ovov', 'oovv', 'ovvo', 'ovoo', 'oooo']:
            coeffs = tuple([mo_coeff_occ, mo_coeff_vir][k == 'v'] for k in key)
            eri1 = mf.with_df.ao2mo(coeffs, compact=False).reshape([c.shape[1] for c in coeffs])
            self.assertAlmostEqual(np.max(np.abs(getattr(eri0, key)-eri1)), 0.0, 10)
        # ovvv and vvvv are compressed with sym=True:
        coeffs = [mo_coeff_occ, mo_coeff_vir, mo_coeff_vir, mo_coeff_vir]
        eri1 = mf.with_df.ao2mo(coeffs, compact=True)
        eri1 = eri1.reshape([coeffs[0].shape[1], coeffs[1].shape[1], -1])
        self.assertAlmostEqual(np.max(np.abs(eri0.ovvv-eri1)), 0.0, 10)
        coeffs = [mo_coeff_vir, mo_coeff_vir, mo_coeff_vir, mo_coeff_vir]
        eri1 = mf.with_df.ao2mo(coeffs, compact=True)
        self.assertAlmostEqual(np.max(np.abs(eri0.vvvv-eri1)), 0.0, 10)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
