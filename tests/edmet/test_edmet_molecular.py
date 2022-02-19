import unittest
import numpy as np
from vayesta import lattmod, edmet
import pyscf.gto
import pyscf.scf
import pyscf.tools.ring


class MolecularEDMETTest:
    ''' Abstract base class for molecular EDMET tests.
    '''

    @classmethod
    def setUpClass(cls):
        cls.mol = None
        cls.mf = None
        cls.edmet = None
        cls.known_values = None

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.edmet, cls.known_values

    def test_energy(self):
        self.assertAlmostEqual(self.edmet.e_tot, self.known_values['e_tot'], 6)


class MolecularEDMETTest_H6_sto6g_EBFCI_IAO_1occ(unittest.TestCase, MolecularEDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.gto.Mole()
        cls.mol.atom = [('H %f %f %f' % xyz) for xyz in pyscf.tools.ring.make(6, 1.0)]
        cls.mol.basis = 'sto-6g'
        cls.mol.build()

        cls.mf = pyscf.scf.RHF(cls.mol)
        cls.mf.kernel()

        cls.edmet = edmet.EDMET(cls.mf, solver='EBFCI', bos_occ_cutoff=1)
        cls.edmet.iao_fragmentation()
        for i in range(cls.mol.natm//2):
            cls.edmet.add_atomic_fragment([i * 2, i * 2 + 1])
        cls.edmet.kernel()

        cls.known_values = {'e_tot': -3.258336016231219 }


class MolecularEDMETTest_H6_sto6g_EBFCI_IAO_2occ(unittest.TestCase, MolecularEDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.gto.Mole()
        cls.mol.atom = [('H %f %f %f' % xyz) for xyz in pyscf.tools.ring.make(6, 1.0)]
        cls.mol.basis = 'sto-6g'
        cls.mol.build()

        cls.mf = pyscf.scf.RHF(cls.mol)
        cls.mf.kernel()

        cls.edmet = edmet.EDMET(cls.mf, solver='EBFCI', bos_occ_cutoff=2)
        cls.edmet.iao_fragmentation()
        for i in range(cls.mol.natm//2):
            cls.edmet.add_atomic_fragment([i * 2, i * 2 + 1])
        cls.edmet.kernel()

        cls.known_values = {'e_tot':  -3.259289706885438}


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
