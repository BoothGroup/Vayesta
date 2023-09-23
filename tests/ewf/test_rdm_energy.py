import unittest

import pyscf
import pyscf.cc

import vayesta.ewf
from tests.common import TestCase
from tests import systems


class Test_RHF(TestCase):
    def test(self):
        # RHF
        mf = systems.water_631g.rhf()

        # CCSD
        cc = pyscf.cc.CCSD(mf)
        cc.kernel()

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


class Test_UHF(TestCase):
    def test(self):
        # RHF
        mf = systems.water_cation_631g.uhf()

        # CCSD
        cc = pyscf.cc.UCCSD(mf)
        cc.kernel()

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


#    def test_h2_solid(self):
#
#        #RHF
#        mf = systems.h2_sto3g_331_2d.rhf()
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
