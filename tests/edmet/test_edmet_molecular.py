import unittest
import numpy as np
from vayesta import lattmod, edmet
import pyscf.gto
import pyscf.scf
import pyscf.tools.ring

def make_test_molecular(atoms, basis, solver, fragment_type, bos_occ_cutoff, known_values, fragments = None):

    class EDMETMolecularTests(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.mol = pyscf.gto.Mole()
            cls.mol.atom = atoms
            cls.mol.basis = basis
            cls.mol.build()

            cls.mf = pyscf.scf.RHF(cls.mol)
            cls.mf.kernel()

            cls.edmet = edmet.EDMET(cls.mf, solver=solver, fragment_type=fragment_type, bos_occ_cutoff = bos_occ_cutoff)
            # Ensure that we don't spam with output.
            cls.edmet.log.setLevel(50)
            if fragments is None:
                cls.edmet.make_all_atom_fragments()
            else:
                for x in fragments:
                    cls.edmet.make_atom_fragment(x)
            cls.edmet.kernel()

        @classmethod
        def tearDownClass(cls):
            del cls.mol, cls.mf, cls.edmet

        def test_energy(self):
            self.assertAlmostEqual(self.edmet.e_tot, known_values['e_tot'])

    return EDMETMolecularTests

def make_test_Hring(natom, d, fragsize, *args, maxiter = 20):
    ring = pyscf.tools.ring.make(natom, d)
    atom = [('H %f %f %f' % xyz) for xyz in ring]
    fragments = [list(range(x, x+fragsize)) for x in range(0, natom, fragsize)]
    return make_test_molecular(atom, *args, fragments = fragments)

edmet_Hring_sto6g_EBFCI_1occ_IAO_nocc_Test = make_test_Hring(6, 1.0, 2, "sto-6g", "EBFCI", "IAO", 1,
                                            {'e_tot': -3.2607921167146703})
edmet_Hring_sto6g_EBFCI_2occ_IAO_nocc_Test = make_test_Hring(6, 1.0, 2, "sto-6g", "EBFCI", "IAO", 2,
                                            {'e_tot': -3.259282652607757})


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()