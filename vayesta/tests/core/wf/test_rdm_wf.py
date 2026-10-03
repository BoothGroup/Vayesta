import pytest
import numpy as np

from vayesta.core.types import SpatialOrbitals
from vayesta.core.types.wf.rdm import RDM_WaveFunction, RRDM_WaveFunction
from vayesta.tests.common import TestCase


@pytest.mark.fast
class TestRRDMWaveFunction(TestCase):
    def make_wf(self, projector=None):
        rng = np.random.default_rng(0)
        norb, nocc = 4, 2
        occ = np.zeros(norb)
        occ[:nocc] = 2
        mo = SpatialOrbitals(rng.random((norb, norb)), energy=rng.random(norb), occ=occ)
        dm1 = rng.random((norb, norb))
        dm2 = rng.random((norb, norb, norb, norb))
        return RDM_WaveFunction(mo, dm1, dm2, projector=projector)

    def test_type(self):
        self.assertIsInstance(self.make_wf(), RRDM_WaveFunction)

    def test_pack_unpack(self):
        for projector in (None, np.eye(4)):
            wf = self.make_wf(projector=projector)
            wf2 = RRDM_WaveFunction.unpack(wf.pack())
            self.assertAllclose(wf2.mo.coeff, wf.mo.coeff)
            self.assertAllclose(wf2.mo.energy, wf.mo.energy)
            self.assertAllclose(wf2.mo.occ, wf.mo.occ)
            self.assertAllclose(wf2.dm1, wf.dm1)
            self.assertAllclose(wf2.dm2, wf.dm2)
            if projector is None:
                self.assertIsNone(wf2.projector)
            else:
                self.assertAllclose(wf2.projector, wf.projector)
