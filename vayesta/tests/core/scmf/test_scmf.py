import unittest

from vayesta import ewf
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems
from vayesta.core.util import cache


class SCMF_Test(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2_dz.rhf()

    @classmethod
    def tearDownClass(cls):
        del cls.mf

    @classmethod
    @cache
    def emb(cls, scmf=None):
        emb = ewf.EWF(cls.mf, solver_options=dict(solve_lambda=True), bath_options=dict(bathtype='dmet'))
        with emb.sao_fragmentation() as f:
            f.add_all_atomic_fragments()
        if scmf == 'pdmet':
            emb.pdmet_scmf()
        elif scmf == 'brueckner':
            emb.brueckner_scmf()
        emb.kernel()
        return emb

    def test_pdmet(self):
        """Test p-DMET."""
        emb0 = self.emb()
        emb = self.emb('pdmet')
        self.assertAllclose(emb.with_scmf.e_tot_oneshot, emb0.e_tot)
        self.assertAllclose(emb.with_scmf.e_tot_oneshot, -1.1419060823155505)
        self.assertTrue(emb.with_scmf.converged)
        self.assertAllclose(emb.with_scmf.e_tot, -1.1417348230969397)

    def test_brueckner(self):
        """Test Brueckner DMET."""
        emb0 = self.emb()
        emb = self.emb('brueckner')
        self.assertAllclose(emb.with_scmf.e_tot_oneshot, emb0.e_tot)
        self.assertAllclose(emb.with_scmf.e_tot_oneshot, -1.1419060823155505)
        self.assertTrue(emb.with_scmf.converged)
        self.assertAllclose(emb.with_scmf.e_tot, -1.1417339799464736)

class SCMF_UHF_Test(SCMF_Test):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2_dz.rhf().to_uhf()


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
