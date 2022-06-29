import unittest
from vayesta.tests.ewf import test_h2
from vayesta.tests import testsystems

class Test_MP2(test_h2.Test_MP2):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_631g.rhf()
        cls.cc = testsystems.water_631g.rmp2()

class Test_CCSD(test_h2.Test_CCSD):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_631g.rhf()
        cls.cc = testsystems.water_631g.rccsd()

class Test_UMP2(test_h2.Test_UMP2):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_cation_631g.uhf()
        cls.cc = testsystems.water_cation_631g.ump2()

class Test_UCCSD(test_h2.Test_UCCSD):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_cation_631g.uhf()
        cls.cc = testsystems.water_cation_631g.uccsd()

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
