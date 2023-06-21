import unittest
import numpy as np

import pyscf
import pyscf.cc

import vayesta
import vayesta.ewf
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems

class Test_RFCI(TestCase):

    def test(self):

        #RHF
        mf = testsystems.h6_sto6g.rhf()

        try:
            from dyson.expressions import FCI
        except ImportError:
            vayesta.log.info("Could not import Dyson. Skipping FCI moment tests.")
            return

        fci = FCI["1h"](mf)
        fci_ip = fci.build_gf_moments(4)

        fci = FCI["1p"](mf)
        fci_ea = fci.build_gf_moments(4)

        #Full bath EWF
        ewf = vayesta.ewf.EWF(mf, bath_type='full', solver_options=dict(n_moments=(4,4)), solver='FCI')
        ewf.kernel()

        for f in ewf.fragments:
            ip, ea = f.results.moms 

            cx = f.get_overlap('mo|cluster')
            ip = np.einsum('pP,qQ,nPQ->npq', cx, cx, ip)
            ea = np.einsum('pP,qQ,nPQ->npq', cx, cx, ea)

            self.assertTrue(np.allclose(ip, fci_ip, atol=1e-14))
            self.assertTrue(np.allclose(ea, fci_ea, atol=1e-14))


class Test_RCCSD(TestCase):

    def test(self):

        #RHF
        mf = testsystems.water_sto3g.rhf()

        try:
            from dyson.expressions import CCSD
        except ImportError:
            vayesta.log.info("Could not import Dyson. Skipping FCI moment tests.")
            return

        cc = CCSD["1h"](mf)
        cc_ip = cc.build_gf_moments(4)

        cc = CCSD["1p"](mf)
        cc_ea = cc.build_gf_moments(4)

        #Full bath EWF
        ewf = vayesta.ewf.EWF(mf, bath_type='full', solver_options=dict(n_moments=(4,4)), solver='CCSD')
        ewf.kernel()

        for f in ewf.fragments:
            ip, ea = f.results.moms 

            cx = f.get_overlap('mo|cluster')
            ip = np.einsum('pP,qQ,nPQ->npq', cx, cx, ip)
            ea = np.einsum('pP,qQ,nPQ->npq', cx, cx, ea)

            for i in range(len(ip)):
                print(i, np.trace(np.dot(ip[i], ip[i])), np.trace(np.dot(cc_ip[i], cc_ip[i])), abs(np.trace(np.dot(ip[i], ip[i])) - np.trace(np.dot(cc_ip[i], cc_ip[i]))))
            print('Norm %f'%np.linalg.norm(ip- cc_ip))
            print('IP %f'%np.linalg.norm(ea- cc_ea))
            self.assertAlmostEqual(np.linalg.norm(ip- cc_ip), 0.0)
            self.assertAlmostEqual(np.linalg.norm(ea- cc_ea), 0.0)

            # self.assertTrue(np.allclose(ip, cc_ip))
            # self.assertTrue(np.allclose(ea, cc_ea))

if __name__ == '__main__':
    print("Running %s" % __file__)
    unittest.main()

