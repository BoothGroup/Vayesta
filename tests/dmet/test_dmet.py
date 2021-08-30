import unittest
import numpy as np
from pyscf import gto, scf
from vayesta import lattmod, dmet

#TODO: full fragmentation ab initio tests are pretty expensive, can we do some init_guess stuff?
#       -> check sum(x.trace() for x in qemb.hl_rdms) == mol.nelectron


def make_mol_tests(atom, basis, known_values):

    class MoleDMETTests(unittest.TestCase):

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
            
        @classmethod
        def tearDownClass(cls):
            del cls.mol, cls.mf

        def test_iao_atoms_ccsd(self):
            qemb = dmet.DMET(self.mf, solver='CCSD', fragment_type='iao', strict=True)
            qemb.make_atom_fragment([0])
            qemb.conv_tol = 1e-6
            qemb.max_elec_error = 1e-4
            qemb.kernel()
            self.assertAlmostEqual(qemb.e_tot, known_values['iao_atoms_ccsd'], 5)

        def test_lowdin_aos_ccsdt(self):
            qemb = dmet.DMET(self.mf, solver='CCSD(T)', fragment_type='lowdin-ao', strict=True)
            qemb.make_ao_fragment([0, 1])
            qemb.conv_tol = 1e-6
            qemb.max_elec_error = 1e-4
            qemb.kernel()
            self.assertAlmostEqual(qemb.e_tot, known_values['lowdin_aos_ccsdt'], 5)

    return MoleDMETTests


def make_lattice_tests(n, nelectron, U, known_values):

    class LatticeDMETTests(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.mol = lattmod.Hubbard1D(n, hubbard_u=U, nelectron=nelectron)
            cls.mf = lattmod.LatticeMF(cls.mol)
            cls.mf.kernel()

        @classmethod
        def tearDownClass(cls):
            del cls.mol, cls.mf

        def test_fci_pairs(self):
            for charge_consistent in [False, True]:
                qemb = dmet.DMET(self.mf, solver='FCI', fragment_type='site', strict=True, charge_consistent=charge_consistent)
                assert not n%2
                frag = qemb.make_atom_fragment([0, 1])
                frags = [frag,] + frag.make_tsymmetric_fragments([n//2, 1, 1])
                qemb.conv_tol = 1e-6
                qemb.max_elec_error = 1e-4
                qemb.kernel()
                self.assertAlmostEqual(qemb.e_tot, known_values['fci_pairs'], 5)
                #TODO: should this be how hl_rdms behaves:
                #self.assertAlmostEqual(sum(x.trace() for x in qemb.hl_rdms), nelectron, 4)
                self.assertAlmostEqual(qemb.hl_rdms[0].trace(), (n*2//nelectron), 4)


    return LatticeDMETTests


LiH_631g_DMETTests = make_mol_tests(
        'Li 0 0 0; H 0 0 1.4',
        'cc-pvdz',
        {
            'iao_atoms_ccsd': -6.388451806770724,
            'lowdin_aos_ccsdt': -6.213660391384988,
        },
)

N10_U0_DMETTests = make_lattice_tests(
        10, 10, 0.0,
        {
            'fci_pairs': -12.944271909999154,
        },
)

N20_U0_DMETTests = make_lattice_tests(
        20, 20, 0.0,
        {
            'fci_pairs': -25.569812885998644,
        },
)

N10_U4_DMETTests = make_lattice_tests(
        10, 10, 4.0,
        {
            'fci_pairs': -5.982495317262652,
        },
)

N10_U8_DMETTests = make_lattice_tests(
        10, 10, 8.0,
        {
            'fci_pairs': -3.3633614776094047,
        },
)
        


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
