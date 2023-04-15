import unittest

from vayesta import ewf
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class HubbardEWFTests(TestCase):
    PLACES_ENERGY = 6
    CONV_TOL = None  #FIXME

    def _test_energy(self, emb, known_values):
        """Tests the EWF energy.
        """

        self.assertAlmostEqual(emb.e_tot, known_values['e_tot'], self.PLACES_ENERGY)

    def test_6_u0_1imp(self):
        """Tests for N=6 U=0 Hubbard model with single site impurities.
        """

        emb = ewf.EWF(
                testsystems.hubb_6_u0.rhf(),
                bno_threshold=1e-8,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                },
        )
        with emb.site_fragmentation() as f:
            frag = f.add_atomic_fragment(0)
        frag.add_tsymmetric_fragments(tvecs=[6, 1, 1])
        emb.kernel()

        known_values = {'e_tot': -8.0}

        self._test_energy(emb, known_values)

    def test_10_u2_2imp(self):
        """Tests for N=10 U=2 Hubbard model with double site impurities.
        """

        emb = ewf.EWF(
                testsystems.hubb_10_u2.rhf(),
                bno_threshold=1e-8,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                },
        )
        with emb.site_fragmentation() as f:
            frag = f.add_atomic_fragment([0, 1])
        frag.add_tsymmetric_fragments(tvecs=[5, 1, 1])
        emb.kernel()

        known_values = {'e_tot': -8.633958869633286}

        self._test_energy(emb, known_values)

    def test_10_u2_2imp_uhf(self):
        """Tests for N=10 U=2 Hubbard model with double site impurities, with a uhf reference.
        """

        emb = ewf.EWF(
                testsystems.hubb_10_u2.rhf(),
                bno_threshold=1e-2,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                },
               solver="FCI"
        )
        with emb.site_fragmentation() as f:
            frag = f.add_atomic_fragment([0, 1])
        frag.add_tsymmetric_fragments(tvecs=[5, 1, 1])
        emb.kernel()


        uemb = ewf.EWF(
                testsystems.hubb_10_u2.uhf(),
                bno_threshold=1e-2,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                },
               solver="FCI"
        )
        with uemb.site_fragmentation() as f:
            frag = f.add_atomic_fragment([0, 1])
        frag.add_tsymmetric_fragments(tvecs=[5, 1, 1])
        uemb.kernel()

        known_values = {'e_tot': emb.e_tot}

        self._test_energy(uemb, known_values)

    def test_6x6_u0_1x1imp(self):
        """Tests for 6x6 U=0 Hubbard model with single site impurities.
        """

        emb = ewf.EWF(
                testsystems.hubb_6x6_u0_1x1imp.rhf(),
                bno_threshold=1e-8,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                },
        )
        with emb.site_fragmentation() as f:
            frag = f.add_atomic_fragment([0])
        frag.add_tsymmetric_fragments(tvecs=[6, 6, 1])
        emb.kernel()

        known_values = {'e_tot': -56.0}

        self._test_energy(emb, known_values)

    def test_6x6_u6_1x1imp(self):
        """Tests for 6x6 U=6 Hubbard model with single site impurities.
        """

        emb = ewf.EWF(
                testsystems.hubb_6x6_u6_1x1imp.rhf(),
                bno_threshold=1e-8,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                },
        )
        with emb.site_fragmentation() as f:
            frag = f.add_atomic_fragment([0])
        frag.add_tsymmetric_fragments(tvecs=[6, 6, 1])
        emb.kernel()

        known_values = {'e_tot': -37.71020224582783}

        self._test_energy(emb, known_values)

    def test_8x8_u2_2x2imp(self):
        """Tests for 8x8 U=2 Hubbard model with 2x2 impurities.
        """

        emb = ewf.EWF(
                testsystems.hubb_8x8_u2_2x2imp.rhf(),
                bno_threshold=1e-8,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                },
        )
        with emb.site_fragmentation() as f:
            frag = f.add_atomic_fragment([0, 1, 2, 3])
        frag.add_tsymmetric_fragments(tvecs=[4, 4, 1])
        emb.kernel()

        known_values = {'e_tot': -84.3268698533661}

        self._test_energy(emb, known_values)



if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
