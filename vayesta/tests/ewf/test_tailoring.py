import unittest
import numpy as np
import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


class TestRestricted(TestCase):

    correction_type = 'tailor'

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.n2_sto_150pm.rhf()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, bno_threshold, tailoring, correction_type=None):
        correction_type = correction_type or cls.correction_type
        solver_opts = dict(conv_tol= 1e-12, conv_tol_normt=1e-8, solve_lambda=False)
        emb = vayesta.ewf.EWF(cls.mf, bath_options=dict(threshold=bno_threshold), solver_options=solver_opts)
        with emb.iao_fragmentation() as f:
            fci1 = f.add_atomic_fragment(0, solver='FCI', bath_options=dict(bathtype='dmet'), auxiliary=True)
            fci2 = f.add_atomic_fragment(1, solver='FCI', bath_options=dict(bathtype='dmet'), auxiliary=True)
            ccsd1 = f.add_atomic_fragment(0, solver='extCCSD')
            ccsd2 = f.add_atomic_fragment(1, solver='extCCSD')
        if tailoring == 'onsite':
            ccsd1.add_external_corrections([fci1], correction_type, projectors=0)
            ccsd2.add_external_corrections([fci2], correction_type, projectors=0)
        elif tailoring == 'all-1p':
            ccsd1.add_external_corrections([fci1, fci2], correction_type, projectors=1)
            ccsd2.add_external_corrections([fci1, fci2], correction_type, projectors=1)
        elif tailoring == 'all-2p':
            ccsd1.add_external_corrections([fci1, fci2], correction_type, projectors=2)
            ccsd2.add_external_corrections([fci1, fci2], correction_type, projectors=2)
        else:
            raise ValueError
        emb.kernel()
        return emb

    def test_wf_energy_onsite(self):
        emb = self.emb(1e-4, 'onsite')
        self.assertAllclose(emb.e_corr, -0.43332813177760987, atol=1e-6, rtol=0)

    def test_dm_energy_onsite(self):
        emb = self.emb(1e-4, 'onsite')
        self.assertAllclose(emb.get_dm_corr_energy(), -0.4073623273252255, atol=1e-6, rtol=0)

    def test_wf_energy_all(self):
        emb = self.emb(1e-4, 'all-1p')
        self.assertAllclose(emb.e_corr, -0.43328577969028964, atol=1e-6, rtol=0)

    def test_dm_energy_all(self):
        emb = self.emb(1e-4, 'all-1p')
        self.assertAllclose(emb.get_dm_corr_energy(), -0.4067136384165535, atol=1e-6, rtol=0)

    def test_wf_energy_all_2p(self):
        emb = self.emb(1e-4, 'all-2p')
        self.assertAllclose(emb.e_corr, -0.4216611412243197, atol=1e-6, rtol=0)

    def test_dm_energy_all_2p(self):
        emb = self.emb(1e-4, 'all-2p')
        self.assertAllclose(emb.get_dm_corr_energy(), -0.4109557775015299, atol=1e-6, rtol=0)


class TestUnrestrictedS0(TestRestricted):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.n2_sto_150pm.rhf().to_uhf()


class TestUnrestrictedS4(TestRestricted):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.n2_sto_s4_150pm.uhf()

    def test_wf_energy_onsite(self):
        emb = self.emb(1e-4, 'onsite')
        self.assertAllclose(emb.e_corr, -0.20294262284592862, atol=1e-6, rtol=0)

    def test_dm_energy_onsite(self):
        emb = self.emb(1e-4, 'onsite')
        self.assertAllclose(emb.get_dm_corr_energy(), -0.2187131950991378, atol=1e-6, rtol=0)

    def test_wf_energy_all(self):
        emb = self.emb(1e-4, 'all-1p')
        self.assertAllclose(emb.e_corr, -0.20279235812388702, atol=1e-6, rtol=0)

    def test_dm_energy_all(self):
        emb = self.emb(1e-4, 'all-1p')
        self.assertAllclose(emb.get_dm_corr_energy(), -0.21845089572304, atol=1e-6, rtol=0)

    def test_wf_energy_all_2p(self):
        emb = self.emb(1e-4, 'all-2p')
        self.assertAllclose(emb.e_corr, -0.2025278277300585, atol=1e-6, rtol=0)

    def test_dm_energy_all_2p(self):
        emb = self.emb(1e-4, 'all-2p')
        self.assertAllclose(emb.get_dm_corr_energy(), -0.21849821198673974, atol=1e-6, rtol=0)


@unittest.skip("Not Implemented")
class TestExternalCorrection(TestRestricted):
    """TODO: replace test values, once implemented."""

    correction_type = 'external'

    def test_wf_energy_onsite(self):
        emb = self.emb(1e-4, 'onsite')
        self.assertAllclose(emb.e_corr, -0.43332813177760987, atol=1e-6, rtol=0)

    def test_dm_energy_onsite(self):
        emb = self.emb(1e-4, 'onsite')
        self.assertAllclose(emb.get_dm_corr_energy(), -0.4073623273252255, atol=1e-6, rtol=0)

    def test_wf_energy_all(self):
        emb = self.emb(1e-4, 'all-1p')
        self.assertAllclose(emb.e_corr, -0.43328577969028964, atol=1e-6, rtol=0)

    def test_dm_energy_all(self):
        emb = self.emb(1e-4, 'all-1p')
        self.assertAllclose(emb.get_dm_corr_energy(), -0.4067136384165535, atol=1e-6, rtol=0)

    def test_wf_energy_all_2p(self):
        emb = self.emb(1e-4, 'all-2p')
        self.assertAllclose(emb.e_corr, -0.4216611412243197, atol=1e-6, rtol=0)

    def test_dm_energy_all_2p(self):
        emb = self.emb(1e-4, 'all-2p')
        self.assertAllclose(emb.get_dm_corr_energy(), -0.4109557775015299, atol=1e-6, rtol=0)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
