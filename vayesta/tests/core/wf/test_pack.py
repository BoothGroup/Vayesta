import pytest
import numpy as np

from vayesta.core.helper import pack_arrays, unpack_arrays
from vayesta.core.types import SpatialOrbitals, SpinOrbitals
from vayesta.core.types.wf.ccsd import RCCSD_WaveFunction, UCCSD_WaveFunction
from vayesta.core.types.wf.mp2 import RMP2_WaveFunction, UMP2_WaveFunction
from vayesta.tests.common import TestCase


NORB = 5
NOCCA, NOCCB = 3, 2


def _random_spatial_orbitals(rng):
    occ = np.zeros(NORB)
    occ[:NOCCA] = 2
    return SpatialOrbitals(rng.random((NORB, NORB)), energy=rng.random(NORB), occ=occ)


def _random_spin_orbitals(rng):
    occa, occb = np.zeros(NORB), np.zeros(NORB)
    occa[:NOCCA] = 1
    occb[:NOCCB] = 1
    coeff = (rng.random((NORB, NORB)), rng.random((NORB, NORB)))
    energy = (rng.random(NORB), rng.random(NORB))
    return SpinOrbitals(coeff, energy=energy, occ=(occa, occb))


def _random_t2_unrestricted(rng):
    nvira, nvirb = NORB - NOCCA, NORB - NOCCB
    return (
        rng.random((NOCCA, NOCCA, nvira, nvira)),
        rng.random((NOCCA, NOCCB, nvira, nvirb)),
        rng.random((NOCCB, NOCCB, nvirb, nvirb)),
    )


@pytest.mark.fast
class TestPack(TestCase):
    def assert_orbitals_equal(self, mo1, mo2):
        self.assertAllclose(mo1.coeff, mo2.coeff)
        self.assertAllclose(mo1.energy, mo2.energy)
        self.assertAllclose(mo1.occ, mo2.occ)

    def assert_optional_allclose(self, a, b):
        if a is None:
            self.assertIsNone(b)
        else:
            self.assertAllclose(a, b)

    def test_pack_arrays(self):
        rng = np.random.default_rng(4)
        arrays = [
            rng.random((3, 4)),
            None,
            np.arange(5),
            rng.random(2) + 1j * rng.random(2),
            rng.random((2, 1, 3)).astype(complex),
            np.zeros(0),
        ]
        unpacked = unpack_arrays(pack_arrays(*arrays))
        self.assertEqual(len(unpacked), len(arrays))
        for a, b in zip(arrays, unpacked):
            if a is None:
                self.assertIsNone(b)
                continue
            self.assertEqual(a.dtype, b.dtype)
            self.assertEqual(a.shape, b.shape)
            self.assertAllclose(a, b)

    def test_rccsd(self):
        rng = np.random.default_rng(0)
        nvir = NORB - NOCCA
        t1 = rng.random((NOCCA, nvir))
        t2 = rng.random((NOCCA, NOCCA, nvir, nvir))
        for with_lambda in (False, True):
            for projector in (None, rng.random((NOCCA, NOCCA))):
                l1, l2 = (t1, t2) if with_lambda else (None, None)
                wf = RCCSD_WaveFunction(
                    _random_spatial_orbitals(rng),
                    t1,
                    t2,
                    l1=l1,
                    l2=l2,
                    projector=projector,
                )
                wf2 = RCCSD_WaveFunction.unpack(wf.pack())
                self.assert_orbitals_equal(wf.mo, wf2.mo)
                self.assertAllclose(wf2.t1, wf.t1)
                self.assertAllclose(wf2.t2, wf.t2)
                self.assert_optional_allclose(wf.l1, wf2.l1)
                self.assert_optional_allclose(wf.l2, wf2.l2)
                self.assert_optional_allclose(wf.projector, wf2.projector)

    def test_uccsd(self):
        rng = np.random.default_rng(1)
        t1 = (rng.random((NOCCA, NORB - NOCCA)), rng.random((NOCCB, NORB - NOCCB)))
        t2 = _random_t2_unrestricted(rng)
        for with_lambda in (False, True):
            for projector in (
                None,
                (rng.random((NOCCA, NOCCA)), rng.random((NOCCB, NOCCB))),
            ):
                l1, l2 = (t1, t2) if with_lambda else (None, None)
                wf = UCCSD_WaveFunction(
                    _random_spin_orbitals(rng),
                    t1,
                    t2,
                    l1=l1,
                    l2=l2,
                    projector=projector,
                )
                wf2 = UCCSD_WaveFunction.unpack(wf.pack())
                self.assert_orbitals_equal(wf.mo, wf2.mo)
                self.assertAllclose(wf2.t1, wf.t1)
                for block in ("t2aa", "t2ab", "t2ba", "t2bb"):
                    self.assertAllclose(getattr(wf2, block), getattr(wf, block))
                if with_lambda:
                    self.assertAllclose(wf2.l1, wf.l1)
                    for block in ("l2aa", "l2ab", "l2ba", "l2bb"):
                        self.assertAllclose(getattr(wf2, block), getattr(wf, block))
                else:
                    self.assertIsNone(wf2.l1)
                    self.assertIsNone(wf2.l2)
                self.assert_optional_allclose(wf.projector, wf2.projector)

    def test_rmp2(self):
        rng = np.random.default_rng(2)
        nvir = NORB - NOCCA
        t2 = rng.random((NOCCA, NOCCA, nvir, nvir))
        for projector in (None, rng.random((NOCCA, NOCCA))):
            wf = RMP2_WaveFunction(_random_spatial_orbitals(rng), t2, projector=projector)
            wf2 = RMP2_WaveFunction.unpack(wf.pack())
            self.assert_orbitals_equal(wf.mo, wf2.mo)
            self.assertAllclose(wf2.t2, wf.t2)
            self.assert_optional_allclose(wf.projector, wf2.projector)

    def test_ump2(self):
        rng = np.random.default_rng(3)
        t2 = _random_t2_unrestricted(rng)
        for projector in (
            None,
            (rng.random((NOCCA, NOCCA)), rng.random((NOCCB, NOCCB))),
        ):
            wf = UMP2_WaveFunction(_random_spin_orbitals(rng), t2, projector=projector)
            wf2 = UMP2_WaveFunction.unpack(wf.pack())
            self.assert_orbitals_equal(wf.mo, wf2.mo)
            for block in ("t2aa", "t2ab", "t2ba", "t2bb"):
                self.assertAllclose(getattr(wf2, block), getattr(wf, block))
            self.assert_optional_allclose(wf.projector, wf2.projector)
