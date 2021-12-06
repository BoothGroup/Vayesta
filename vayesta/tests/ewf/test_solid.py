import unittest
import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.cc

from vayesta import ewf
from vayesta.tests.cache import cells


class SolidEWFTests(unittest.TestCase):

    def test_lih(self):
        # Mean-field
        kmf = cells['lih_k221']['rhf']
        gmf = cells['lih_g221']['rhf']
        nk = len(kmf.kpts)
        self.assertAlmostEqual(kmf.e_tot, gmf.e_tot/nk)
        # PySCF
        kccsd = pyscf.pbc.cc.KCCSD(kmf)
        kccsd.kernel()
        # k-points
        kemb = ewf.EWF(kmf, bno_threshold=-1)
        kemb.kernel()
        # G-point
        gemb = ewf.EWF(gmf, bno_threshold=-1)
        gemb.iao_fragmentation()
        gemb.add_atomic_fragment(0, sym_factor=nk)
        gemb.add_atomic_fragment(1, sym_factor=nk)
        gemb.kernel()

        self.assertAlmostEqual(kemb.e_tot, -8.069261598354077)
        self.assertAlmostEqual(gemb.e_tot/nk, -8.069261598354077)
        self.assertAlmostEqual(kccsd.e_tot, -8.069261598354077)

    def test_boron(self):
        """Odd electron, UHF test"""

        # Mean-field
        kmf = cells['boron_cp_k321']['uhf']
        gmf = cells['boron_cp_g321']['uhf']
        nk = len(kmf.kpts)
        self.assertAlmostEqual(kmf.e_tot, gmf.e_tot/nk)
        # Do not compare to KUCCSD:
        # PySCF KUCCSD and KRCCSD are different by the exxdiv correction (this is a PySCF bug)
        #kccsd = pyscf.pbc.cc.KUCCSD(kmf)
        #kccsd.kernel()
        # k-points
        kemb = ewf.EWF(kmf, bno_threshold=-1)
        kemb.kernel()
        # G-point
        gemb = ewf.EWF(gmf, bno_threshold=-1)
        gemb.iao_fragmentation()
        gemb.add_atomic_fragment(0, sym_factor=nk)
        gemb.kernel()

        self.assertAlmostEqual(kemb.e_tot, -24.405747542914185)
        self.assertAlmostEqual(gemb.e_tot/nk, -24.405747542914185)
        #self.assertAlmostEqual(kccsd.e_tot, -24.154337069693625)

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
