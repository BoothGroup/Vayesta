import pytest

from vayesta.misc.gto_helper import make_counterpoise_fragments
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


@pytest.mark.fast
class TestCounterpoiseFragments(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = testsystems.water_sto3g.mol

    def test_full_basis(self):
        fmols = make_counterpoise_fragments(self.mol, [["O"]], dump_input=False)
        # Oxygen fragment and remaining hydrogen atoms:
        self.assertEqual(len(fmols), 2)
        for fmol in fmols:
            self.assertEqual(fmol.natm, self.mol.natm)
            self.assertEqual(fmol.nao, self.mol.nao)
        self.assertEqual(fmols[0].nelectron, 8)
        self.assertEqual(fmols[1].nelectron, 2)

    def test_fragment_basis(self):
        fmols = make_counterpoise_fragments(
            self.mol,
            [["O"], ["H"]],
            full_basis=False,
            add_rest_fragment=False,
            dump_input=False,
        )
        self.assertEqual(len(fmols), 2)
        self.assertEqual(fmols[0].natm, 1)
        self.assertEqual(fmols[1].natm, 2)
        self.assertEqual(fmols[0].nao + fmols[1].nao, self.mol.nao)

    def test_invalid_fragment(self):
        with self.assertRaises(ValueError):
            make_counterpoise_fragments(self.mol, [["Li"]], dump_input=False)
