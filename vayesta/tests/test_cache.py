import unittest
import numpy as np

from vayesta.tests.cache import moles, allowed_keys_mole
from vayesta.tests.cache import cells, allowed_keys_cell
from vayesta.tests.cache import latts, allowed_keys_latt


class MolTests:
    key = None
    mf_key = 'rhf'
    cache = moles
    known_values = {
    }
    ENERGY_PLACES = 10
    MATRIX_PLACES = 6
    STABLE_PLACES = 10

    @classmethod
    def tearDownClass(cls):
        del cls.known_values

    def test_converged(self):
        """Test convergence.
        """

        self.assertTrue(self.cache[self.key][self.mf_key].converged)

    def test_e_tot(self):
        """Test the HF energy.
        """

        e_tot = self.cache[self.key][self.mf_key].e_tot
        self.assertAlmostEqual(e_tot, self.known_values['e_tot'], self.ENERGY_PLACES)

    def test_canonical(self):
        """Test that the reference is canonical.
        """

        f1 = np.linalg.multi_dot((
            self.cache[self.key][self.mf_key].mo_coeff.T,
            self.cache[self.key][self.mf_key].get_fock(),
            self.cache[self.key][self.mf_key].mo_coeff,
        ))
        f2 = np.diag(self.cache[self.key][self.mf_key].mo_energy)

        self.assertAlmostEqual(np.max(np.abs(f1-f2)), 0.0, self.MATRIX_PLACES)

        d1 = np.linalg.multi_dot((
            self.cache[self.key][self.mf_key].mo_coeff.T,
            self.cache[self.key][self.mf_key].get_ovlp(),
            self.cache[self.key][self.mf_key].make_rdm1(),
            self.cache[self.key][self.mf_key].get_ovlp(),
            self.cache[self.key][self.mf_key].mo_coeff,
        ))
        d2 = np.diag(self.cache[self.key][self.mf_key].mo_occ)

        self.assertAlmostEqual(np.max(np.abs(d1-d2)), 0.0, self.MATRIX_PLACES)

    def test_stability(self):
        """Test that the mean-field solution is stable.
        """

        mo0 = self.cache[self.key][self.mf_key].mo_coeff
        mo0 = np.array(mo0)

        mo1 = self.cache[self.key][self.mf_key].stability()[0]
        mo1 = np.array(mo1)

        self.assertAlmostEqual(np.max(np.abs(mo0-mo1)), 0.0, self.STABLE_PLACES)


class UMolTests(MolTests):
    mf_key = 'uhf'
    PLACES_SPIN = 10

    def test_canonical(self):
        """Test that the reference is canonical.
        """

        for i in range(2):
            f1 = np.linalg.multi_dot((
                self.cache[self.key][self.mf_key].mo_coeff[i].T,
                self.cache[self.key][self.mf_key].get_fock()[i],
                self.cache[self.key][self.mf_key].mo_coeff[i],
            ))
            f2 = np.diag(self.cache[self.key][self.mf_key].mo_energy[i])

            self.assertAlmostEqual(np.max(np.abs(f1-f2)), 0.0, self.MATRIX_PLACES)

            d1 = np.linalg.multi_dot((
                self.cache[self.key][self.mf_key].mo_coeff[i].T,
                self.cache[self.key][self.mf_key].get_ovlp(),
                self.cache[self.key][self.mf_key].make_rdm1()[i],
                self.cache[self.key][self.mf_key].get_ovlp(),
                self.cache[self.key][self.mf_key].mo_coeff[i],
            ))
            d2 = np.diag(self.cache[self.key][self.mf_key].mo_occ[i])

            self.assertAlmostEqual(np.max(np.abs(d1-d2)), 0.0, self.MATRIX_PLACES)

    def test_spin_squared(self):
        """Test the expectation value of S^2.
        """

        s2 = self.cache[self.key][self.mf_key].spin_square()[0]

        self.assertAlmostEqual(s2, 0.0, self.PLACES_SPIN)


class CellTests(MolTests):
    cache = cells

    def test_canonical(self):
        """Test that the reference is canonical.
        """

        fock = self.cache[self.key][self.mf_key].get_fock()

        for k in range(len(self.cache[self.key]['kpts'])):
            f1 = np.linalg.multi_dot((
                self.cache[self.key][self.mf_key].mo_coeff[k].T,
                fock[k],
                self.cache[self.key][self.mf_key].mo_coeff[k],
            ))
            f2 = np.diag(self.cache[self.key][self.mf_key].mo_energy[k])

            self.assertAlmostEqual(np.max(np.abs(f1-f2)), 0.0, self.MATRIX_PLACES)

            d1 = np.linalg.multi_dot((
                self.cache[self.key][self.mf_key].mo_coeff[k].T,
                self.cache[self.key][self.mf_key].get_ovlp()[k],
                self.cache[self.key][self.mf_key].make_rdm1()[k],
                self.cache[self.key][self.mf_key].get_ovlp()[k],
                self.cache[self.key][self.mf_key].mo_coeff[k],
            ))
            d2 = np.diag(self.cache[self.key][self.mf_key].mo_occ[k])

            self.assertAlmostEqual(np.max(np.abs(d1-d2)), 0.0, self.MATRIX_PLACES)

    test_stability = None


class UCellTests(CellTests, UMolTests):
    def test_canonical(self):
        """Test that the reference is canonical.
        """

        fock = self.cache[self.key][self.mf_key].get_fock()

        for i in range(2):
            for k in range(len(self.cache[self.key]['kpts'])):
                f1 = np.linalg.multi_dot((
                    self.cache[self.key][self.mf_key].mo_coeff[i][k].T,
                    fock[i][k],
                    self.cache[self.key][self.mf_key].mo_coeff[i][k],
                ))
                f2 = np.diag(self.cache[self.key][self.mf_key].mo_energy[i][k])

                self.assertAlmostEqual(np.max(np.abs(f1-f2)), 0.0, self.MATRIX_PLACES)

                d1 = np.linalg.multi_dot((
                    self.cache[self.key][self.mf_key].mo_coeff[i][k].T,
                    self.cache[self.key][self.mf_key].get_ovlp()[k],
                    self.cache[self.key][self.mf_key].make_rdm1()[i][k],
                    self.cache[self.key][self.mf_key].get_ovlp()[k],
                    self.cache[self.key][self.mf_key].mo_coeff[i][k],
                ))
                d2 = np.diag(self.cache[self.key][self.mf_key].mo_occ[i][k])

                self.assertAlmostEqual(np.max(np.abs(d1-d2)), 0.0, self.MATRIX_PLACES)

    test_stability = None


class LattTests(MolTests):
    cache = latts


# RHF molecules:

class h2_ccpvdz_rhf_Tests(unittest.TestCase, MolTests):
    key = 'h2_ccpvdz'
    known_values = {'e_tot': -1.1001537648784097}

class h2o_ccpvdz_rhf_Tests(unittest.TestCase, MolTests):
    key = 'h2o_ccpvdz'
    known_values = {'e_tot': -76.0267720533941}

class h2o_ccpvdz_df_rhf_Tests(unittest.TestCase, MolTests):
    key = 'h2o_ccpvdz_df'
    known_values = {'e_tot': -76.0267511405444}

class n2_631g_rhf_Tests(unittest.TestCase, MolTests):
    key = 'n2_631g'
    known_values = {'e_tot': -108.8676183730583}

class n2_ccpvdz_df_rhf_Tests(unittest.TestCase, MolTests):
    key = 'n2_ccpvdz_df'
    known_values = {'e_tot': -108.95348837904693}

class lih_ccpvdz_rhf_Tests(unittest.TestCase, MolTests):
    key = 'lih_ccpvdz'
    known_values = {'e_tot': -7.9763539426740895}

class h6_sto6g_rhf_Tests(unittest.TestCase, MolTests):
    key = 'h6_sto6g'
    known_values = {'e_tot': -3.1775491323759173}

class h10_sto6g_rhf_Tests(unittest.TestCase, MolTests):
    key = 'h10_sto6g'
    known_values = {'e_tot': -5.27545180026029}


# UHF molecules:

class h2_ccpvdz_uhf_Tests(unittest.TestCase, UMolTests):
    key = 'h2_ccpvdz'
    known_values = {'e_tot': -1.1001537648784097}

class h2o_ccpvdz_uhf_Tests(unittest.TestCase, UMolTests):
    key = 'h2o_ccpvdz'
    known_values = {'e_tot': -76.02677205339408}

class h2o_ccpvdz_df_uhf_Tests(unittest.TestCase, UMolTests):
    key = 'h2o_ccpvdz_df'
    known_values = {'e_tot': -76.02675114054428}

class n2_631g_uhf_Tests(unittest.TestCase, UMolTests):
    key = 'n2_631g'
    known_values = {'e_tot': -108.86761837305833}

class n2_ccpvdz_df_uhf_Tests(unittest.TestCase, UMolTests):
    key = 'n2_ccpvdz_df'
    known_values = {'e_tot': -108.95348837904693}

class lih_ccpvdz_uhf_Tests(unittest.TestCase, UMolTests):
    key = 'lih_ccpvdz'
    known_values = {'e_tot': -7.976353942674095}

class h6_sto6g_uhf_Tests(unittest.TestCase, UMolTests):
    key = 'h6_sto6g'
    known_values = {'e_tot': -3.177549132375921}

class h10_sto6g_uhf_Tests(unittest.TestCase, UMolTests):
    key = 'h10_sto6g'
    known_values = {'e_tot': -5.27545180026029}


# RHF solids:

class he2_631g_222_rhf_Tests(unittest.TestCase, CellTests):
    key = 'he2_631g_222'
    known_values = {'e_tot': -5.711761178431758}

class he_631g_222_rhf_Tests(unittest.TestCase, CellTests):
    key = 'he_631g_222'
    known_values = {'e_tot': -2.8584823308467895}

class h2_sto3g_331_2d_rhf_Tests(unittest.TestCase, CellTests):
    key = 'h2_sto3g_331_2d'
    known_values = {'e_tot': -0.8268798378506185}


# UHF solids:

class he2_631g_222_uhf_Tests(unittest.TestCase, UCellTests):
    key = 'he2_631g_222'
    known_values = {'e_tot': -5.7117611784317575}


# RHF lattices:

class hubb_6_u0_rhf_Tests(unittest.TestCase, LattTests):
    key = 'hubb_6_u0'
    known_values = {'e_tot': -8.0}

class hubb_10_u2_rhf_Tests(unittest.TestCase, LattTests):
    key = 'hubb_10_u2'
    known_values = {'e_tot': -7.9442719099991566}

class hubb_16_u4_rhf_Tests(unittest.TestCase, LattTests):
    key = 'hubb_16_u4'
    known_values = {'e_tot': 17.991547869638765}

class hubb_6x6_u0_1x1imp_rhf_Tests(unittest.TestCase, LattTests):
    key = 'hubb_6x6_u0_1x1imp'
    known_values = {'e_tot': -56.0}

class hubb_6x6_u2_1x1imp_rhf_Tests(unittest.TestCase, LattTests):
    key = 'hubb_6x6_u2_1x1imp'
    known_values = {'e_tot': -46.6111111111112}

class hubb_6x6_u6_1x1imp_rhf_Tests(unittest.TestCase, LattTests):
    key = 'hubb_6x6_u6_1x1imp'
    known_values = {'e_tot': -27.8333333333334}

class hubb_8x8_u2_2x2imp_rhf_Tests(unittest.TestCase, LattTests):
    key = 'hubb_8x8_u2_2x2imp'
    known_values = {'e_tot': -81.72358399593925}

class hubb_8x8_u2_2x1imp_rhf_Tests(unittest.TestCase, LattTests):
    key = 'hubb_8x8_u2_2x1imp'
    known_values = {'e_tot': -81.72358399593925}


# Check they're all there:

for key in allowed_keys_mole:
    for mf_key in ['rhf', 'uhf']:
        if moles[key][mf_key] is not False:
            assert ('%s_%s_Tests' % (key, mf_key)) in globals(), ('%s_%s_Tests' % (key, mf_key))

for key in allowed_keys_cell:
    for mf_key in ['rhf', 'uhf']:
        if cells[key][mf_key] is not False:
            assert ('%s_%s_Tests' % (key, mf_key)) in globals(), ('%s_%s_Tests' % (key, mf_key))

for key in allowed_keys_latt:
    for mf_key in ['rhf', 'uhf']:
        if latts[key][mf_key] is not False:
            assert ('%s_%s_Tests' % (key, mf_key)) in globals(), ('%s_%s_Tests' % (key, mf_key))


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
