import unittest
from unittest import mock

import pyscf

from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


class Test_UHF_Stable(TestCase):
    def test_linear_dependency_in_stability(self):
        testsystems.h2_sto3g_dissoc.uhf_stable.cache_clear()
        mf = testsystems.h2_sto3g_dissoc.uhf()

        with mock.patch.object(
            type(mf),
            "stability",
            side_effect=pyscf.lib.exceptions.LinearDependencyError("Initial guess is empty or zero"),
        ):
            stable_mf = testsystems.h2_sto3g_dissoc.uhf_stable()

        self.assertTrue(stable_mf.converged)


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
