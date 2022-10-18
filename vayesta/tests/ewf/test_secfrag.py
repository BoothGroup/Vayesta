import unittest
import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


class Test_Restricted(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_631g.rhf()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, bno_threshold):
        emb = vayesta.ewf.EWF(cls.mf, bath_options=dict(threshold=bno_threshold))
        with emb.fragmentation() as f:
            with f.secondary_fragments(solver='MP2', bno_threshold_factor=0.1):
                f.add_all_atomic_fragments()
        emb.kernel()
        return emb

    def test_dmet_bath(self):
        emb = self.emb(100)
        self.assertAllclose(emb.e_tot, -76.02292750136847)

    def test_bno_bath(self):
        emb = self.emb(1e-3)
        self.assertAllclose(emb.e_tot, -76.10732883213784)

    def test_full_bath(self):
        emb = self.emb(-1)
        self.assertAllclose(emb.e_tot, -76.11935398347251)

class Test_RestrictedRotsym(Test_Restricted):
    """Test secondary fragments in combination with rotational symmetry."""

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h6_dz.rhf()

    @classmethod
    @cache
    def emb_nosym(cls, bno_threshold):
        emb = vayesta.ewf.EWF(cls.mf, bath_options=dict(threshold=bno_threshold))
        with emb.sao_fragmentation() as f:
            with f.secondary_fragments(solver='MP2', bno_threshold_factor=0.1):
                f.add_all_atomic_fragments()
        emb.kernel()
        return emb

    @classmethod
    @cache
    def emb_sym(cls, bno_threshold):
        emb = vayesta.ewf.EWF(cls.mf, bath_options=dict(threshold=bno_threshold))
        emb.symmetry.add_rotation(order=6, axis=(0,0,1), center=(0,0,0))
        with emb.sao_fragmentation() as f:
            with f.secondary_fragments(solver='MP2', bno_threshold_factor=0.1):
                f.add_atomic_fragment(0)
        emb.kernel()
        return emb

    def test_dmet_bath(self):
        e_ref = -3.2821414000373434
        self.assertAllclose(self.emb_sym(100).e_tot, e_ref)
        self.assertAllclose(self.emb_nosym(100).e_tot, e_ref)

    def test_bno_bath(self):
        e_ref = -3.3803355754550233
        self.assertAllclose(self.emb_sym(1e-3).e_tot, e_ref)
        self.assertAllclose(self.emb_nosym(1e-3).e_tot, e_ref)

    def test_full_bath(self):
        e_ref = -3.3844029599490364
        self.assertAllclose(self.emb_sym(-1).e_tot, e_ref)
        self.assertAllclose(self.emb_nosym(-1).e_tot, e_ref)


class Test_Unrestricted(Test_Restricted):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_cation_631g.uhf()

    def test_dmet_bath(self):
        emb = self.emb(100)
        self.assertAllclose(emb.e_tot, -75.6125617745999)

    def test_bno_bath(self):
        emb = self.emb(1e-3)
        self.assertAllclose(emb.e_tot, -75.67397715335292)

    def test_full_bath(self):
        emb = self.emb(-1)
        self.assertAllclose(emb.e_tot, -75.6830484961464)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()