import unittest
import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.cc

from vayesta import ewf
from vayesta.tests.cache import cells


class SolidEWFTests(unittest.TestCase):

    def test_restricted(self):
        """Tests restriced EWF for a solid, binary system with and without k-point folding."""
        # Mean-field
        kmf = cells['lih_k221']['rhf']  # Primitive cell and k-points
        gmf = cells['lih_g221']['rhf']  # Supercell and Gamma-point
        nk = len(kmf.kpts)
        self.assertAlmostEqual(kmf.e_tot, gmf.e_tot/nk)
        # PySCF
        kccsd = pyscf.pbc.cc.KCCSD(kmf)
        # k-points
        kemb = ewf.EWF(kmf)
        # G-point
        gemb = ewf.EWF(gmf)
        gemb.iao_fragmentation()
        gemb.add_atomic_fragment(0, sym_factor=nk)
        gemb.add_atomic_fragment(1, sym_factor=nk)

        # --- Test full bath
        e_expected = -8.069261598354077
        kemb.kernel(bno_threshold=-1)
        gemb.kernel(bno_threshold=-1)
        kccsd.kernel()
        self.assertAlmostEqual(kemb.e_tot, e_expected)
        self.assertAlmostEqual(gemb.e_tot/nk, e_expected)
        self.assertAlmostEqual(kccsd.e_tot, e_expected)

        # --- Test partial bath
        e_expected = -8.068896307452492
        kemb.kernel(bno_threshold=1e-5)
        gemb.kernel(bno_threshold=1e-5)
        self.assertAlmostEqual(kemb.e_tot, e_expected)
        self.assertAlmostEqual(gemb.e_tot/nk, e_expected)

    def test_restricted_2d(self):
        """Tests restriced EWF for a 2D solid system with and without k-point folding."""
        # Mean-field
        kmf = cells['graphene_k221']['rhf']  # Primitive cell and k-points
        gmf = cells['graphene_g221']['rhf']  # Supercell and Gamma-point
        nk = len(kmf.kpts)
        self.assertAlmostEqual(kmf.e_tot, gmf.e_tot/nk)
        # PySCF
        kccsd = pyscf.pbc.cc.KCCSD(kmf)
        # k-points
        kemb = ewf.EWF(kmf)
        kemb.iao_fragmentation()
        kemb.add_atomic_fragment(0, sym_factor=2)
        # G-point
        gemb = ewf.EWF(gmf)
        gemb.iao_fragmentation()
        gemb.add_atomic_fragment(0, sym_factor=2*nk)

        # --- Test full bath
        e_expected = -75.89352268881753
        kemb.kernel(bno_threshold=-1)
        gemb.kernel(bno_threshold=-1)
        kccsd.kernel()
        self.assertAlmostEqual(kemb.e_tot, e_expected)
        self.assertAlmostEqual(gemb.e_tot/nk, e_expected, places=6)
        self.assertAlmostEqual(kccsd.e_tot, e_expected)

        # --- Test partial bath
        e_expected = -75.8781119002179
        kemb.kernel(bno_threshold=1e-5)
        gemb.kernel(bno_threshold=1e-5)
        self.assertAlmostEqual(kemb.e_tot, e_expected)
        self.assertAlmostEqual(gemb.e_tot/nk, e_expected, places=6)

    def test_unrestricted(self):
        """Tests unrestriced EWF for a solid, odd electron system with and without k-point folding."""
        # Mean-field
        kmf = cells['boron_cp_k321']['uhf'] # Primitive cell and k-points
        gmf = cells['boron_cp_g321']['uhf'] # Supercell and Gamma-point
        nk = len(kmf.kpts)
        self.assertAlmostEqual(kmf.e_tot, gmf.e_tot/nk)
        # Do not compare to KUCCSD
        # PySCF KUCCSD and KRCCSD differ by the exxdiv correction (this is a PySCF bug in KUCCSD)
        #kccsd = pyscf.pbc.cc.KUCCSD(kmf)
        # k-points
        kemb = ewf.EWF(kmf)
        # G-point
        gemb = ewf.EWF(gmf)
        gemb.iao_fragmentation()
        gemb.add_atomic_fragment(0, sym_factor=nk)

        # --- Test full bath
        kemb.kernel(bno_threshold=-1)
        gemb.kernel(bno_threshold=-1)
        #kccsd.kernel()
        e_expected = -24.405747542914185
        self.assertAlmostEqual(kemb.e_tot, e_expected)
        self.assertAlmostEqual(gemb.e_tot/nk, e_expected)
        #self.assertAlmostEqual(kccsd.e_tot, -24.154337069693625)

        # --- Test partial bath
        e_expected = -24.40568870697553
        kemb.kernel(bno_threshold=1e-5)
        gemb.kernel(bno_threshold=1e-5)
        self.assertAlmostEqual(kemb.e_tot, e_expected)
        self.assertAlmostEqual(gemb.e_tot/nk, e_expected)

    def test_unrestricted_2d(self):
        """Tests unrestriced EWF for a solid, odd electron system with and without k-point folding."""
        # Mean-field
        kmf = cells['hydrogen_cubic_2d_k221']['uhf'] # Primitive cell and k-points
        gmf = cells['hydrogen_cubic_2d_g221']['uhf'] # Supercell and Gamma-point
        nk = len(kmf.kpts)
        self.assertAlmostEqual(kmf.e_tot, gmf.e_tot/nk)
        # Do not compare to KUCCSD
        # PySCF KUCCSD and KRCCSD differ by the exxdiv correction (this is a PySCF bug in KUCCSD)
        #kccsd = pyscf.pbc.cc.KUCCSD(kmf)
        # k-points
        kemb = ewf.EWF(kmf)
        # G-point
        gemb = ewf.EWF(gmf)
        gemb.iao_fragmentation()
        gemb.add_atomic_fragment(0, sym_factor=nk)

        # --- Test full bath
        kemb.kernel(bno_threshold=-1)
        gemb.kernel(bno_threshold=-1)
        #kccsd.kernel()
        e_expected = -0.45610319689117035
        self.assertAlmostEqual(kemb.e_tot, e_expected)
        self.assertAlmostEqual(gemb.e_tot/nk, e_expected)
        #self.assertAlmostEqual(kccsd.e_tot, e_expected)

        # --- Test partial bath
        e_expected = -0.45551437750186297
        kemb.kernel(bno_threshold=1e-5)
        gemb.kernel(bno_threshold=1e-5)
        self.assertAlmostEqual(kemb.e_tot, e_expected)
        self.assertAlmostEqual(gemb.e_tot/nk, e_expected)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
