import pytest

from vayesta.core.qemb import Embedding
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


@pytest.mark.fast
class TestFragmentCounterpoise(TestCase):
    def test_make_counterpoise_mol(self):
        mf = testsystems.water_sto3g.rhf()
        emb = Embedding(mf)
        with emb.iao_fragmentation() as f:
            frag = f.add_atomic_fragment(0)
            frag2 = f.add_atomic_fragment([1, 2])
        mol_cp = frag.make_counterpoise_mol(rmax=10.0)
        self.assertEqual(mol_cp.natm, mf.mol.natm)
        self.assertEqual(mol_cp.atom_symbol(0), "O")
        self.assertEqual(mol_cp.nelectron, 8)
        mol_cp = frag.make_counterpoise_mol(rmax=0)
        self.assertEqual(mol_cp.natm, 1)
        with self.assertRaises(NotImplementedError):
            frag2.make_counterpoise_mol(rmax=10.0)
