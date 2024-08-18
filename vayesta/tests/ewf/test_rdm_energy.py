import unittest

import pyscf
import pyscf.cc

import vayesta.ewf
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class Test_RHF(TestCase):
    def test_h2o(self):
        # RHF
        mf = testsystems.water_631g.rhf()

        # CCSD
        cc = testsystems.water_631g.rccsd()

        # Full bath EWF
        ewf = vayesta.ewf.EWF(mf, bath_options=dict(bathtype="full"), solver_options=dict(solve_lambda=True))
        ewf.kernel()

        ll = ewf._get_dm_corr_energy_old(global_dm1=False, global_dm2=False)
        gl = ewf._get_dm_corr_energy_old(global_dm1=True, global_dm2=False)
        lg = ewf._get_dm_corr_energy_old(global_dm1=False, global_dm2=True)
        gg = ewf._get_dm_corr_energy_old(global_dm1=True, global_dm2=True)

        self.assertAlmostEqual(ll, cc.e_corr)
        self.assertAlmostEqual(gl, cc.e_corr)
        self.assertAlmostEqual(lg, cc.e_corr)
        self.assertAlmostEqual(gg, cc.e_corr)
        self.assertAlmostEqual(ewf.get_dm_energy(), cc.e_tot)

    def test_h2_solid(self):

        #RHF
        mf = testsystems.h2_sto3g_k311.rhf()

        #CCSD
        cc = testsystems.h2_sto3g_k311.rccsd()

        #Full bath EWF
        ewf = vayesta.ewf.EWF(mf, bath_options=dict(bathtype="full"), solver_options=dict(solve_lambda=True))
        ewf.kernel()

        ll = ewf._get_dm_corr_energy_old(global_dm1=False, global_dm2=False)
        gl = ewf._get_dm_corr_energy_old(global_dm1=True, global_dm2=False)
        lg = ewf._get_dm_corr_energy_old(global_dm1=False, global_dm2=True)
        gg = ewf._get_dm_corr_energy_old(global_dm1=True, global_dm2=True)

        self.assertAlmostEqual(ll, cc.e_corr)
        self.assertAlmostEqual(gl, cc.e_corr)
        self.assertAlmostEqual(lg, cc.e_corr)
        self.assertAlmostEqual(gg, cc.e_corr)


class Test_UHF(TestCase):
    def test_h2o(self):
        # RHF
        mf = testsystems.water_cation_631g.uhf()

        # CCSD
        cc = testsystems.water_cation_631g.uccsd()
        # Full bath EWF
        ewf = vayesta.ewf.EWF(mf, bath_options=dict(bathtype="full"), solver_options=dict(solve_lambda=True))
        ewf.kernel()

        ll = ewf._get_dm_corr_energy_old(global_dm1=False, global_dm2=False)
        gl = ewf._get_dm_corr_energy_old(global_dm1=True, global_dm2=False)
        lg = ewf._get_dm_corr_energy_old(global_dm1=False, global_dm2=True)
        gg = ewf._get_dm_corr_energy_old(global_dm1=True, global_dm2=True)

        self.assertAlmostEqual(ll, cc.e_corr)
        self.assertAlmostEqual(gl, cc.e_corr)
        self.assertAlmostEqual(lg, cc.e_corr)
        self.assertAlmostEqual(gg, cc.e_corr)
        self.assertAlmostEqual(ewf.get_dm_energy(), cc.e_tot)

    # EXXDIV maybe?
    # def test_h2_solid(self):

    #     #RHF
    #     mf = testsystems.h2_sto3g_k311.uhf()

    #     #CCSD
    #     cc = testsystems.h2_sto3g_k311.uccsd()

    #     #Full bath EWF
    #     ewf = vayesta.ewf.EWF(mf, bath_options=dict(bathtype="full"), solver_options=dict(solve_lambda=True))
    #     ewf.kernel()

    #     ll = ewf._get_dm_corr_energy_old(global_dm1=False, global_dm2=False)
    #     gl = ewf._get_dm_corr_energy_old(global_dm1=True, global_dm2=False)
    #     lg = ewf._get_dm_corr_energy_old(global_dm1=False, global_dm2=True)
    #     gg = ewf._get_dm_corr_energy_old(global_dm1=True, global_dm2=True)

    #     self.assertAlmostEqual(ll, cc.e_corr)
    #     self.assertAlmostEqual(gl, cc.e_corr)
    #     self.assertAlmostEqual(lg, cc.e_corr)
    #     self.assertAlmostEqual(gg, cc.e_corr)


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
