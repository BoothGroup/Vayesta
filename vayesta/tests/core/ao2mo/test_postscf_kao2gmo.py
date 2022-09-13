import pytest
import unittest
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems
from vayesta.core.util import replace_attr
from vayesta.core.ao2mo import postscf_kao2gmo
from vayesta.core.ao2mo import postscf_kao2gmo_uhf


@pytest.mark.slow
class Test_CCSD(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2_sto3g_k311.rhf()
        cls.cc = testsystems.h2_sto3g_s311.rccsd()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        del cls.cc

    def test_postscf_kao2gmo(self):
        eris_ref = self.cc.ao2mo()
        kmf = self.mf
        gmf = self.cc._scf
        with replace_attr(gmf, exxdiv=None):
            fock = gmf.get_fock()
            e_hf = gmf.energy_tot()
        eris = postscf_kao2gmo(self.cc, kmf.with_df, fock=fock, mo_energy=gmf.mo_energy, e_hf=e_hf)
        self.assertAllclose(eris.oooo.flatten(), eris_ref.oooo.flatten())
        self.assertAllclose(eris.ovoo.flatten(), eris_ref.ovoo.flatten())
        self.assertAllclose(eris.ovvo.flatten(), eris_ref.ovvo.flatten())
        self.assertAllclose(eris.oovv.flatten(), eris_ref.oovv.flatten())
        self.assertAllclose(eris.ovov.flatten(), eris_ref.ovov.flatten())
        self.assertAllclose(eris.ovvv.flatten(), eris_ref.ovvv.flatten())
        self.assertAllclose(eris.vvvv.flatten(), eris_ref.vvvv.flatten())
        self.assertAllclose(eris.fock, eris_ref.fock)
        self.assertAllclose(eris.mo_energy, eris_ref.mo_energy)
        # PySCF version < v2.1:
        if hasattr(eris, 'e_hf'):
            self.assertAllclose(eris.e_hf, eris_ref.e_hf)

@pytest.mark.slow
class Test_UCCSD(Test_CCSD):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h3_sto3g_k311.uhf()
        cls.cc = testsystems.h3_sto3g_s311.uccsd()

    def test_postscf_kao2gmo(self):
        eris_ref = self.cc.ao2mo()
        kmf = self.mf
        gmf = self.cc._scf
        with replace_attr(gmf, exxdiv=None):
            fock = gmf.get_fock()
            e_hf = gmf.energy_tot()
        eris = postscf_kao2gmo_uhf(self.cc, kmf.with_df, fock=fock, mo_energy=gmf.mo_energy, e_hf=e_hf)
        # Alpha-Alpha
        self.assertAllclose(eris.oooo.flatten(), eris_ref.oooo.flatten())
        self.assertAllclose(eris.ovoo.flatten(), eris_ref.ovoo.flatten())
        self.assertAllclose(eris.ovvo.flatten(), eris_ref.ovvo.flatten())
        self.assertAllclose(eris.oovv.flatten(), eris_ref.oovv.flatten())
        self.assertAllclose(eris.ovov.flatten(), eris_ref.ovov.flatten())
        # TODO: Currently (ov|vv) and (vv|vv) integrals are not tested, as they need to be unpacked
        #self.assertAllclose(eris.ovvv.flatten(), eris_ref.ovvv.flatten())
        #self.assertAllclose(eris.vvvv.flatten(), eris_ref.vvvv.flatten())
        # Alpha-Beta
        self.assertAllclose(eris.ooOO.flatten(), eris_ref.ooOO.flatten())
        self.assertAllclose(eris.ovOO.flatten(), eris_ref.ovOO.flatten())
        self.assertAllclose(eris.ovVO.flatten(), eris_ref.ovVO.flatten())
        self.assertAllclose(eris.ooVV.flatten(), eris_ref.ooVV.flatten())
        self.assertAllclose(eris.ovOV.flatten(), eris_ref.ovOV.flatten())
        #self.assertAllclose(eris.ovVV.flatten(), eris_ref.ovVV.flatten())
        #self.assertAllclose(eris.vvVV.flatten(), eris_ref.vvVV.flatten())
        # Beta-Alpha
        self.assertAllclose(eris.OVoo.flatten(), eris_ref.OVoo.flatten())
        self.assertAllclose(eris.OVvo.flatten(), eris_ref.OVvo.flatten())
        self.assertAllclose(eris.OOvv.flatten(), eris_ref.OOvv.flatten())
        #self.assertAllclose(eris.OVvv.flatten(), eris_ref.OVvv.flatten())
        # Beta-Beta
        self.assertAllclose(eris.OOOO.flatten(), eris_ref.OOOO.flatten())
        self.assertAllclose(eris.OVOO.flatten(), eris_ref.OVOO.flatten())
        self.assertAllclose(eris.OVVO.flatten(), eris_ref.OVVO.flatten())
        self.assertAllclose(eris.OOVV.flatten(), eris_ref.OOVV.flatten())
        self.assertAllclose(eris.OVOV.flatten(), eris_ref.OVOV.flatten())
        #self.assertAllclose(eris.OVVV.flatten(), eris_ref.OVVV.flatten())
        #self.assertAllclose(eris.VVVV.flatten(), eris_ref.VVVV.flatten())
        # Other
        self.assertAllclose(eris.fock, eris_ref.fock)
        self.assertAllclose(eris.mo_energy, eris_ref.mo_energy)
        # PySCF version < v2.1:
        if hasattr(eris, 'e_hf'):
            self.assertAllclose(eris.e_hf, eris_ref.e_hf)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
