import unittest
import numpy as np

import pyscf
import pyscf.cc

import vayesta.ewf
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems

class TestRDMEnergyConvergence(TestCase):

    def test_water(self):

        #RHF
        mf = testsystems.h2o_ccpvdz.rhf()

        #CCSD
        cc = pyscf.cc.CCSD(mf)
        cc.kernel()

        #Full bath EWF
        ewf = vayesta.ewf.EWF(mf, bath_type='full', solve_lambda=True)
        ewf.kernel()

        ll = ewf.get_dm_corr_energy_old(global_dm1=False, global_dm2=False)
        gl = ewf.get_dm_corr_energy_old(global_dm1=True, global_dm2=False)
        lg = ewf.get_dm_corr_energy_old(global_dm1=False, global_dm2=True)
        gg = ewf.get_dm_corr_energy_old(global_dm1=True, global_dm2=True)

        self.assertAlmostEqual(ll, cc.e_corr)
        self.assertAlmostEqual(gl, cc.e_corr)
        self.assertAlmostEqual(lg, cc.e_corr)
        self.assertAlmostEqual(gg, cc.e_corr)


#    def test_h2_solid(self):
#
#        #RHF
#        mf = testsystems.h2_sto3g_331_2d.rhf()
#
#        #CCSD
#        cc = pyscf.cc.CCSD(mf)
#        cc.kernel()
#
#        #Full bath EWF
#        ewf = vayesta.ewf.EWF(mf, bno_threshold=-1)
#        ewf.kernel()
#
#        ll = ewf.get_rdm2_corr_energy(global_dm1=False, global_dm2=False)
#        gl = ewf.get_rdm2_corr_energy(global_dm1=True, global_dm2=False)
#        lg = ewf.get_rdm2_corr_energy(global_dm1=False, global_dm2=True)
#        gg = ewf.get_rdm2_corr_energy(global_dm1=True, global_dm2=True)
#
#        self.assertAlmostEqual(ll, cc.e_corr)
#        self.assertAlmostEqual(gl, cc.e_corr)
#        self.assertAlmostEqual(lg, cc.e_corr)
#        self.assertAlmostEqual(gg, cc.e_corr)

if __name__ == '__main__':
    print("Running %s" % __file__)
    unittest.main()
