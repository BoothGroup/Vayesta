import unittest
import numpy as np
from vayesta import lattmod, dmet
import pyscf.gto
import pyscf.scf
import pyscf.tools.ring


class MolecularDMETTest:
    ''' Abstract base class for molecular DMET tests.
    '''

    @classmethod
    def setUpClass(cls):
        cls.mol = None
        cls.mf = None
        cls.dmet = None
        cls.known_values = None

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.dmet, cls.known_values

    def test_converged(self):
        self.assertTrue(self.dmet.converged)

    def test_energy(self):
        self.assertAlmostEqual(self.dmet.e_tot, self.known_values['e_tot'], 6)


class MolecularDMETTest_H10_sto6g_FCI_IAO_cc(unittest.TestCase, MolecularDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.gto.Mole()
        cls.mol.atom = [('H %f %f %f' % xyz) for xyz in pyscf.tools.ring.make(10, 1.0)]
        cls.mol.basis = "sto-6g"
        cls.mol.verbose = 0
        cls.mol.build()

        cls.mf = pyscf.scf.RHF(cls.mol)
        cls.mf.kernel()

        cls.dmet = dmet.DMET(
                cls.mf,
                solver='FCI',
                charge_consistent=True,
                bath_type=None,
                fragment_type='IAO',
        )
        for x in range(5):
            cls.dmet.make_atom_fragment([x*2, x*2+1])
        cls.dmet.kernel()

        cls.known_values = {'e_tot': -5.421103410208376}


class MolecularDMETTest_H10_sto6g_FCI_IAO_nocc(unittest.TestCase, MolecularDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.gto.Mole()
        cls.mol.atom = [('H %f %f %f' % xyz) for xyz in pyscf.tools.ring.make(10, 1.0)]
        cls.mol.basis = "sto-6g"
        cls.mol.verbose = 0
        cls.mol.build()

        cls.mf = pyscf.scf.RHF(cls.mol)
        cls.mf.kernel()

        cls.dmet = dmet.DMET(
                cls.mf,
                solver='FCI',
                charge_consistent=False,
                bath_type=None,
                fragment_type='IAO',
        )
        for x in range(5):
            cls.dmet.make_atom_fragment([x*2, x*2+1])
        cls.dmet.kernel()

        cls.known_values = {'e_tot': -5.421192647967002}


class MolecularDMETTest_H10_sto6g_FCI_IAO_all(unittest.TestCase, MolecularDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.gto.Mole()
        cls.mol.atom = [('H %f %f %f' % xyz) for xyz in pyscf.tools.ring.make(10, 1.0)]
        cls.mol.basis = "sto-6g"
        cls.mol.verbose = 0
        cls.mol.build()

        cls.mf = pyscf.scf.RHF(cls.mol)
        cls.mf.kernel()

        cls.dmet = dmet.DMET(
                cls.mf,
                solver='FCI',
                charge_consistent=False,
                bath_type='all',
                fragment_type='IAO',
        )
        for x in range(5):
            cls.dmet.make_atom_fragment([x*2, x*2+1])
        cls.dmet.kernel()

        cls.known_values = {'e_tot': -5.422668582405825}


class MolecularDMETTest_H10_sto6g_FCI_IAO_BNO(unittest.TestCase, MolecularDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.gto.Mole()
        cls.mol.atom = [('H %f %f %f' % xyz) for xyz in pyscf.tools.ring.make(10, 1.0)]
        cls.mol.basis = "sto-6g"
        cls.mol.verbose = 0
        cls.mol.build()

        cls.mf = pyscf.scf.RHF(cls.mol)
        cls.mf.kernel()

        cls.dmet = dmet.DMET(
                cls.mf,
                solver='FCI',
                charge_consistent=False,
                bath_type='MP2-BNO',
                bno_threshold=np.inf,
                fragment_type='IAO',
        )
        for x in range(5):
            cls.dmet.make_atom_fragment([x*2, x*2+1])
        cls.dmet.kernel()

        cls.known_values = {'e_tot': -5.421192648085972}


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
