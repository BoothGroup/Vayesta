import unittest

from vayesta import ewf
from vayesta.tests.cache import moles


class SCMFTests(unittest.TestCase):
    key = 'n2_631g'
    mf_key = 'rhf'
    PLACES = 7
    CONV_TOL_E = 1e-10
    CONV_TOL_D = 1e-8

    def test_pdmet(self):
        """Test p-DMET.
        """

        emb = ewf.EWF(moles[self.key][self.mf_key], make_rdm1=True, bath_type=None)
        emb.sao_fragmentation()
        emb.add_all_atomic_fragments()
        emb.pdmet_scmf(etol=self.CONV_TOL_E, dtol=self.CONV_TOL_D)
        emb.kernel()

        self.assertTrue(emb.with_scmf.converged)
        self.assertAlmostEqual(emb.with_scmf.e_tot_oneshot, -109.06727638760327, self.PLACES)
        self.assertAlmostEqual(emb.with_scmf.e_tot,         -109.06845425753974, self.PLACES)

    def test_brueckner(self):
        """Test Brueckner DMET.
        """

        emb = ewf.EWF(moles[self.key][self.mf_key], make_rdm1=True, bath_type=None)
        emb.sao_fragmentation()
        emb.add_all_atomic_fragments()
        emb.brueckner_scmf(etol=self.CONV_TOL_E, dtol=self.CONV_TOL_D)
        emb.kernel()

        self.assertTrue(emb.with_scmf.converged)
        self.assertAlmostEqual(emb.with_scmf.e_tot_oneshot, -109.06727638760327, self.PLACES)
        self.assertAlmostEqual(emb.with_scmf.e_tot,         -109.06844454464843, self.PLACES)


class USCMFTests(SCMFTests):
    key = 'n2_631g'
    mf_key = 'uhf'
    PLACES = 7


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
