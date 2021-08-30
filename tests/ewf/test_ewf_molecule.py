import unittest
import numpy as np
from pyscf import gto, scf
from vayesta import ewf


def make_test(atom, basis, kwargs, fragmentation, known_values, name=None):

    class MoleculeEWFTests(unittest.TestCase):

        shortDescription = lambda self: name

        @classmethod
        def setUpClass(cls):
            cls.mol = gto.Mole()
            cls.mol.atom = atom
            cls.mol.basis = basis
            cls.mol.verbose = 0
            cls.mol.max_memory = 1e9
            cls.mol.build()
            cls.mf = scf.RHF(cls.mol)
            cls.mf.conv_tol = 1e-12
            cls.mf.kernel()
            cls.ewf = ewf.EWF(cls.mf, solver_options={'conv_tol': 1e-10}, **kwargs)
            fragmentation(cls.ewf)
            cls.ewf.kernel()

        @classmethod
        def tearDownClass(cls):
            del cls.mol, cls.mf, cls.ewf

        def test_energy(self):
            self.assertAlmostEqual(self.ewf.e_tot, known_values['e_tot'], 8)

    return MoleculeEWFTests


LiH_ccpvdz_iao_atoms_Test = make_test(
        'Li 0 0 0; H 0 0 1.4', 'cc-pvdz',
        {'fragment_type': 'iao', 'bath_type': 'all'},
        lambda ewf: ewf.make_all_atom_fragments(),
        {'e_tot': -8.008269603007381},
        name='LiH_ccpvdz_iao_atoms_Test',
)

LiH_ccpvdz_lowdin_aos_Test = make_test(
        'Li 0 0 0; H 0 0 1.4', 'cc-pvdz',
        {'fragment_type': 'lowdin-ao', 'bno_threshold': 1e-5},
        lambda ewf: (ewf.make_ao_fragment([0, 1]), ewf.make_ao_fragment([2, 3, 4])),
        {'e_tot': -7.98424889149862},
        name='LiH_ccpvdz_lowdin_aos_Test',
)

#N2_augccpvdz_stretched_FCI_Test = make_test(
#        'N 0 0 0; N 0 0 2', 'aug-cc-pvdz',
#        {'solver': 'FCI', 'bno_threshold': 100},
#        lambda ewf: ewf.make_atom_fragment(0, sym_factor=2),
#        {'e_tot': -108.7770182190321},
#        name='N2_augccpvdz_stretched_FCI_Test',
#)
#
#N2_ccpvdz_TCCSD_Test = make_test(
#        'N1 0 0 0; N2 0 0 1.1', 'cc-pvdz',
#        {'solver': 'TCCSD', 'bno_threshold': 1e-4},
#        lambda ewf: ewf.make_atom_fragment('N1', sym_factor=2),
#        {'e_tot': -109.27077981413623},
#        name='N2_ccpvdz_TCCSD_Test',
#)

N2_ccpvdz_sc_Test = make_test(
        'N1 0 0 0; N2 0 0 1.1', 'cc-pvdz',
        {'bno_threshold': 1e-4, 'sc_mode': 1, 'sc_energy_tol': 1e-9},
        lambda ewf: ewf.make_atom_fragment(0, sym_factor=2),
        {'e_tot': -109.26013012932687},
        name='N2_ccpvdz_sc_Test',
)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
