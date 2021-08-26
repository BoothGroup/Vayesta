import unittest
import numpy as np
from pyscf.pbc import gto, scf, tools
from vayesta.core.ao2mo import kao2gmo
from vayesta.misc import gdf


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

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kpts, cls.mf

    def test_gdf_to_eris(self):
        mf = tools.k2gamma.k2gamma(self.mf)
        mf = mf.density_fit()
        mo_coeff_occ = mf.mo_coeff[:, mf.mo_occ > 0]
        mo_coeff_vir = mf.mo_coeff[:, mf.mo_occ == 0]
        eri0 = kao2gmo.gdf_to_eris(self.mf.with_df, mf.mo_coeff, nocc=mo_coeff_occ.shape[1], only_ovov=False)
        for key in eri0.keys():
            coeffs = tuple([mo_coeff_occ, mo_coeff_vir][k == 'v'] for k in key)
            eri1 = mf.with_df.ao2mo(coeffs, compact=False).reshape([c.shape[1] for c in coeffs])
            self.assertAlmostEqual(np.max(np.abs(eri0[key]-eri1)), 0.0, 10)



if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
