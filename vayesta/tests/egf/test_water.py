import unittest

from dyson.expressions import FCI, CCSD
from dyson.solvers.chempot import AufbauPrinciple
from vayesta import egf
from vayesta.core.util import AbstractMethodError, cache
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems
from vayesta.tests.egf.test_h6 import Test_Full_Bath_CCSD 

class Test_Full_Bath_CCSD(Test_Full_Bath_CCSD):

    system = testsystems.water_ccpvdz_df

if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()



