import pytest
import unittest
from pyscf.pbc import scf
from vayesta.core import foldscf
from tests.common import TestCase
from tests import testsystems


@pytest.mark.slow
class FoldSCF_RHF_Tests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kmf = testsystems.he2_631g_k222.rhf()
        cls.mf = foldscf.fold_scf(cls.kmf)
        cls.scell = testsystems.he2_631g_s222.mol
        cls.smf = testsystems.he2_631g_s222.rhf()

    @classmethod
    def teardownClass(cls):
        del cls.kmf, cls.mf, cls.scell, cls.smf

    def test_e_tot(self):
        """Compare the HF energies."""
        e_tot = self.kmf.e_tot * len(self.kmf.kpts)
        self.assertAllclose(e_tot, self.mf.e_tot)
        self.assertAllclose(e_tot, self.smf.e_tot)

    def test_ovlp(self):
        """Compare the overlap matrices."""
        self.assertAllclose(self.mf.get_ovlp(), self.smf.get_ovlp())

    def test_hcore(self):
        """Compare the core Hamiltonians."""
        self.assertAllclose(self.mf.get_hcore(), self.smf.get_hcore())

    def test_veff(self):
        """Compare the HF potentials."""
        self.assertAllclose(self.mf.get_veff(), self.smf.get_veff())

    def test_fock(self):
        """Compare the Fock matrices."""
        scell, phase = foldscf.get_phase(self.kmf.mol, self.kmf.kpts)
        dm = foldscf.k2bvk_2d(self.kmf.get_init_guess(), phase)
        self.assertAllclose(self.mf.get_fock(dm=dm), self.smf.get_fock(dm=dm))


@pytest.mark.slow
class FoldSCF_UHF_Tests(FoldSCF_RHF_Tests):
    @classmethod
    def setUpClass(cls):
        cls.kmf = testsystems.he2_631g_k222.uhf()
        cls.mf = foldscf.fold_scf(cls.kmf)
        cls.scell = testsystems.he2_631g_s222.mol
        cls.smf = scf.UHF(cls.scell)
        cls.smf.conv_tol = 1e-12
        cls.smf = cls.smf.density_fit()
        cls.smf.kernel()


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
