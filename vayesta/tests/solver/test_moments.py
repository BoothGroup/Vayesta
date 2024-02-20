import unittest
import pytest
import numpy as np

import pyscf
import pyscf.cc

import vayesta
import vayesta.ewf
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class Test_Spectral_Moments(TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import dyson
        except ImportError:
            pytest.skip("Requires dyson")

    def test_fci(self):
        # RHF
        mf = testsystems.h6_sto6g.rhf()

        from dyson.expressions import FCI

        fci = FCI["1h"](mf)
        fci_ip = fci.build_gf_moments(4)

        fci = FCI["1p"](mf)
        fci_ea = fci.build_gf_moments(4)

        # Full bath EWF
        ewf = vayesta.ewf.EWF(
            mf, bath_options=dict(bathtype="full"), solver_options=dict( n_moments=(4, 4)), solver="FCI"
        )
        ewf.kernel()

        for f in ewf.fragments:
            ip, ea = f.results.moms

            cx = f.get_overlap("mo|cluster")
            ip = np.einsum("pP,qQ,nPQ->npq", cx, cx, ip)
            ea = np.einsum("pP,qQ,nPQ->npq", cx, cx, ea)
            
            self.assertTrue(np.allclose(ip, fci_ip))
            self.assertTrue(np.allclose(ea, fci_ea))

    def test_ccsd(self):

        #RHF
        mf = testsystems.water_sto3g.rhf()

        from dyson.expressions import CCSD

        cc = CCSD["1h"](mf)
        cc_ip = cc.build_gf_moments(4)
          
        cc = CCSD["1p"](mf)
        cc_ea = cc.build_gf_moments(4)

        #Full bath EWF
        ewf = vayesta.ewf.EWF(mf, bath_options=dict(bathtype='full'), solver_options=dict(n_moments=(4,4)), solver='CCSD')
        ewf.kernel()

        for f in ewf.fragments:
            ip, ea = f.results.moms

            cx = f.get_overlap('mo|cluster')
            ip = np.einsum('pP,qQ,nPQ->npq', cx, cx, ip)
            ea = np.einsum('pP,qQ,nPQ->npq', cx, cx, ea)

            # High tolerence for github CI
            self.assertTrue(np.allclose(ip, cc_ip, atol=1e-3))
            self.assertTrue(np.allclose(ea, cc_ea, atol=1e-6))

if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
