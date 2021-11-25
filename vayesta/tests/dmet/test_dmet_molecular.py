import unittest
import numpy as np

import pyscf.gto
import pyscf.scf

from vayesta import dmet
from vayesta.tests.cache import mols

#TODO: increase precision


class MolecularDMETTest:

    key = None
    mf_key = None
    PLACES = 7

    @classmethod
    def setUpClass(cls):
        cls.dmet = None
        cls.known_values = None

    @classmethod
    def tearDownClass(cls):
        del cls.dmet, cls.known_values

    def test_converged(self):
        """Test that the DMET has converged.
        """

        self.assertTrue(self.dmet.converged)

    def test_energy(self):
        """Test that the energy matches a known value.
        """

        self.assertAlmostEqual(self.dmet.e_tot, self.known_values['e_tot'], self.PLACES)


class MolecularDMETTest_H6_sto6g_FCI_IAO_cc(unittest.TestCase, MolecularDMETTest):

    key = 'h6_sto6g'
    mf_key = 'rhf'

    @classmethod
    def setUpClass(cls):
        cls.dmet = dmet.DMET(
                mols[cls.key][cls.mf_key],
                solver='FCI',
                charge_consistent=True,
                bath_type=None,
                conv_tol=1e-9,
        )
        cls.dmet.iao_fragmentation()
        for x in range(3):
            cls.dmet.make_atom_fragment([x*2, x*2+1])
        cls.dmet.kernel()

        cls.known_values = {'e_tot': -3.2596560444286524}


class MolecularDMETTest_H6_sto6g_FCI_IAO_nocc(MolecularDMETTest_H6_sto6g_FCI_IAO_cc):

    @classmethod
    def setUpClass(cls):
        cls.dmet = dmet.DMET(
                mols[cls.key][cls.mf_key],
                solver='FCI',
                charge_consistent=False,
                bath_type=None,
                conv_tol=1e-9,
        )
        cls.dmet.iao_fragmentation()
        for x in range(3):
            cls.dmet.make_atom_fragment([x*2, x*2+1])
        cls.dmet.kernel()

        cls.known_values = {'e_tot': -3.2596414844443995}


class MolecularDMETTest_H6_sto6g_FCI_IAO_all(MolecularDMETTest_H6_sto6g_FCI_IAO_cc):
    @classmethod
    def setUpClass(cls):
        cls.dmet = dmet.DMET(
                mols[cls.key][cls.mf_key],
                solver='FCI',
                charge_consistent=False,
                bath_type='all',
                conv_tol=1e-9,
        )
        cls.dmet.iao_fragmentation()
        for x in range(3):
            cls.dmet.make_atom_fragment([x*2, x*2+1])
        cls.dmet.kernel()

        cls.known_values = {'e_tot': -3.2585986118561703}


class MolecularDMETTest_H6_sto6g_FCI_IAO_BNO(MolecularDMETTest_H6_sto6g_FCI_IAO_cc):
    @classmethod
    def setUpClass(cls):
        cls.dmet = dmet.DMET(
                mols[cls.key][cls.mf_key],
                solver='FCI',
                charge_consistent=False,
                bath_type='MP2-BNO',
                bno_threshold=np.inf,
                conv_tol=1e-9,
        )
        cls.dmet.iao_fragmentation()
        for x in range(3):
            cls.dmet.make_atom_fragment([x*2, x*2+1])
        cls.dmet.kernel()

        cls.known_values = {'e_tot': -3.2596414844443995}


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
