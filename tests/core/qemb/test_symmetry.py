import unittest
import numpy as np
import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from tests.systems import MoleculeTestSystem
from tests.common import TestCase


CENTER = np.asarray((0.1, 0.2, -0.3))
co2_geom = [
    ("C", CENTER),
    ("O1", CENTER + np.asarray([0, 0, -1.163])),
    ("O2", CENTER + np.asarray([0, 0, +1.163])),
]
co2 = MoleculeTestSystem(atom=co2_geom, basis="6-31G", incore_anyway=True)


class TestCO2(TestCase):
    system = co2

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.rhf()
        cls.cc = cls.system.rccsd()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        del cls.cc
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, bno_threshold, symmetry=None, **kwargs):
        emb = vayesta.ewf.EWF(cls.mf, bath_options=dict(threshold=bno_threshold))
        with emb.fragmentation() as f:
            if symmetry is None:
                f.add_all_atomic_fragments()
            elif symmetry == "inversion":
                f.add_atomic_fragment(0)
                with f.inversion_symmetry(center=CENTER):
                    f.add_atomic_fragment(1)
            elif symmetry == "reflection":
                f.add_atomic_fragment(0)
                with f.mirror_symmetry(center=CENTER, **kwargs):
                    f.add_atomic_fragment(1)
            elif symmetry == "rotation":
                f.add_atomic_fragment(0)
                with f.rotational_symmetry(center=CENTER, order=2, **kwargs):
                    f.add_atomic_fragment(1)
        emb.kernel()
        return emb

    def test_inversion(self):
        emb = self.emb(np.inf)
        emb_sym = self.emb(np.inf, symmetry="inversion")
        dm1 = emb.make_rdm1()
        dm1_sym = emb_sym.make_rdm1()
        self.assertAllclose(dm1_sym, dm1)

    def test_reflection(self):
        emb = self.emb(np.inf)
        emb_sym = self.emb(np.inf, symmetry="reflection", axis="z")
        dm1 = emb.make_rdm1()
        dm1_sym = emb_sym.make_rdm1()
        self.assertAllclose(dm1_sym, dm1)

    def test_rotation(self):
        emb = self.emb(np.inf)
        dm1 = emb.make_rdm1()
        # Test different rotation axes
        for ax in ("x", "y", (0.6, 1.3, 0)):
            emb_sym = self.emb(np.inf, symmetry="rotation", axis=ax)
            dm1_sym = emb_sym.make_rdm1()
            self.assertAllclose(dm1_sym, dm1)
