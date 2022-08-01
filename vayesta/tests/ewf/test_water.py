import pytest
import unittest
from vayesta.tests.ewf import test_h2
from vayesta.tests import testsystems

@pytest.mark.slow
class Test_MP2(test_h2.Test_MP2):

    system = testsystems.water_631g

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.rhf()
        cls.cc = cls.system.rmp2()

@pytest.mark.slow
class Test_CCSD(test_h2.Test_CCSD):

    system = testsystems.water_631g

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.rhf()
        cls.cc = cls.system.rccsd()

@pytest.mark.slow
class Test_UMP2(test_h2.Test_UMP2):

    system = testsystems.water_cation_631g

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.uhf()
        cls.cc = cls.system.ump2()

@pytest.mark.slow
class Test_UCCSD(test_h2.Test_UCCSD):

    system = testsystems.water_cation_631g

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.uhf()
        cls.cc = cls.system.uccsd()

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
