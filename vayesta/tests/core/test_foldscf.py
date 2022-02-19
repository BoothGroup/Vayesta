import unittest
import numpy as np

from pyscf.pbc import scf, tools

from vayesta.core import foldscf
from vayesta.tests.cache import cells


class FoldSCF_RHF_Tests(unittest.TestCase):
    key = 'he2_631g_222'
    mf_key = 'rhf'
    GMF = staticmethod(scf.RHF)
    PLACES_ENERGY = 10
    PLACES_MATRIX = 8

    @classmethod
    def setUpClass(cls):
        cls.mf = foldscf.fold_scf(cells[cls.key][cls.mf_key])
        kmesh = tools.k2gamma.kpts_to_kmesh(cells[cls.key]['cell'], cells[cls.key]['kpts'])
        cls.scell = tools.super_cell(cells[cls.key]['cell'], kmesh)
        cls.smf = cls.GMF(cls.scell)
        cls.smf.conv_tol = 1e-12
        cls.smf = cls.smf.density_fit()
        cls.smf.kernel()

    @classmethod
    def teardownClass(cls):
        del cls.mf, cls.scell, cls.smf

    def _test_values(self, a, b, prec=8):
        self.assertAlmostEqual(np.abs(np.max(a-b)), 0.0, prec)

    def test_e_tot(self):
        """Compare the HF energies.
        """

        e_tot = cells[self.key][self.mf_key].e_tot * len(cells[self.key]['kpts'])
        self.assertAlmostEqual(e_tot, self.mf.e_tot, self.PLACES_ENERGY)
        self.assertAlmostEqual(e_tot, self.smf.e_tot, self.PLACES_ENERGY)

    def test_ovlp(self):
        """Compare the overlap matrices.
        """

        self._test_values(self.mf.get_ovlp(), self.smf.get_ovlp(), self.PLACES_MATRIX)

    def test_hcore(self):
        """Compare the core Hamiltonians.
        """

        self._test_values(self.mf.get_hcore(), self.smf.get_hcore(), self.PLACES_MATRIX)

    def test_veff(self):
        """Compare the HF potentials.
        """

        self._test_values(self.mf.get_veff(), self.smf.get_veff(), self.PLACES_MATRIX)

    def test_fock(self):
        """Compare the Fock matrices.
        """

        scell, phase = foldscf.get_phase(cells[self.key]['cell'], cells[self.key]['kpts'])
        dm = foldscf.k2bvk_2d(cells[self.key][self.mf_key].get_init_guess(), phase)
        self._test_values(self.mf.get_fock(dm=dm), self.smf.get_fock(dm=dm), self.PLACES_MATRIX)


class FoldSCF_UHF_Tests(FoldSCF_RHF_Tests):
    mf_key = 'uhf'
    GMF = staticmethod(scf.UHF)

    def _test_values(self, a, b, prec=8):
        if isinstance(a, (tuple, list)):
            for _a, _b in zip(a, b):
                self.assertAlmostEqual(np.abs(np.max(_a-_b)), 0.0, prec)
        else:
            self.assertAlmostEqual(np.abs(np.max(a-b)), 0.0, prec)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
