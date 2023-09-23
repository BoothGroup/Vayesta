import unittest
import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from tests import systems
from tests.common import TestCase


class Test_Restricted(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mf = systems.water_631g.rhf()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, bno_threshold):
        emb = vayesta.ewf.EWF(cls.mf, bath_options=dict(threshold=bno_threshold, project_dmet_order=0))
        emb.kernel()
        return emb

    def test_full_bath(self):
        emb = self.emb(-1)
        self.assertAllclose(emb.get_fbc_energy(), 0)

    def test_finite_bath(self):
        emb = self.emb(1e-3)
        self.assertAllclose(emb.get_fbc_energy(), -0.00965818665787619)

    def test_finite_bath_occ(self):
        emb = self.emb(1e-3)
        self.assertAllclose(emb.get_fbc_energy(virtual=False), -0.000749028286918895)

    def test_finite_bath_vir(self):
        emb = self.emb(1e-3)
        self.assertAllclose(emb.get_fbc_energy(occupied=False), -0.00890915837083031)


class Test_Unrestricted(Test_Restricted):
    @classmethod
    def setUpClass(cls):
        cls.mf = systems.water_cation_631g.uhf()

    def test_finite_bath(self):
        emb = self.emb(1e-3)
        self.assertAllclose(emb.get_fbc_energy(), -0.009390484069477173)

    def test_finite_bath_occ(self):
        emb = self.emb(1e-3)
        self.assertAllclose(emb.get_fbc_energy(virtual=False), -0.0005469048120275064)

    def test_finite_bath_vir(self):
        emb = self.emb(1e-3)
        self.assertAllclose(emb.get_fbc_energy(occupied=False), -0.008843579257449625)
