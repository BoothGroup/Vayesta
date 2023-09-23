import pytest
import unittest
from tests.ewf import test_h2
from tests import systems


@pytest.mark.slow
class Test_MP2(test_h2.Test_MP2):
    system = systems.water_631g

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.rhf()
        cls.cc = cls.system.rmp2()


@pytest.mark.slow
class Test_CCSD(test_h2.Test_CCSD):
    system = systems.water_631g

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.rhf()
        cls.cc = cls.system.rccsd()


@pytest.mark.slow
class Test_UMP2(test_h2.Test_UMP2):
    system = systems.water_cation_631g

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.uhf()
        cls.cc = cls.system.ump2()


@pytest.mark.slow
class Test_UCCSD(test_h2.Test_UCCSD):
    system = systems.water_cation_631g

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.uhf()
        cls.cc = cls.system.uccsd()
