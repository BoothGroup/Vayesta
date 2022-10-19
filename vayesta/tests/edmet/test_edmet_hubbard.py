import unittest

from vayesta import edmet
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class EDMET_Hubbard_Tests(TestCase):
    PLACES_ENERGY = 6

    def _test_energy(self, emb, known_values):
        """Test that the energy matches a known value.
        """
        self.assertAlmostEqual(emb.e_tot, known_values['e_tot'], self.PLACES_ENERGY)

    #FIXME bug #9
    #def test_6_u0_1imp_1occ(self):
    #    """Tests for N=6 U=0 Hubbard model with single site impurities.
    #    """

    #    emb = edmet.EDMET(
    #            testsystems.hubb_6_u0.rhf(),
    #            solver='FCI',
    #            max_boson_occ=1,
    #    )
    #    emb.site_fragmentation()
    #    frag = emb.add_atomic_fragment(0)
    #    frag.add_tsymmetric_fragments(tvecs=[6, 1, 1])
    #    emb.kernel()

    #    known_values = {'e_tot': -8.0}

    #    self._test_energy(emb, known_values)

    #def test_6_u0_2imp_6occ(self):
    #    """Tests for N=6 U=0 Hubbard model with double site impurities.
    #    """

    #    emb = edmet.EDMET(
    #            testsystems.hubb_6_u0.rhf(),
    #            solver='FCI',
    #            max_boson_occ=6,
    #    )
    #    emb.site_fragmentation()
    #    frag = emb.add_atomic_fragment([0, 1])
    #    frag.add_tsymmetric_fragments(tvecs=[3, 1, 1])
    #    emb.kernel()

    #    known_values = {'e_tot': -8.0}

    #    self._test_energy(emb, known_values)

    def test_10_u2_2imp_2occ(self):
        """Tests for N=10 U=2 Hubbard model with double site impurities.
        """

        emb = edmet.EDMET(
                testsystems.hubb_10_u2.rhf(),
                solver='FCI',
                solver_options={"max_boson_occ": 2},
                bosonic_interaction="direct",
                oneshot=True,
                make_dd_moments=False,
        )
        with emb.site_fragmentation() as f:
            frag = f.add_atomic_fragment([0, 1])
            frag.add_tsymmetric_fragments(tvecs=[5, 1, 1])
        emb.kernel()

        known_values = {'e_tot':-8.793485086132375}

        self._test_energy(emb, known_values)

    # Suspended this test in favour of one with EBCCSD due to time constraints.
    #def test_14_upoint4_2imp_4occ(self):
    #    """Tests for N=14 U=0.4 Hubbard model with double site impurities.
    #    """

    #    emb = edmet.EDMET(
    #            testsystems.hubb_14_u0.rhf(),
    #            solver='FCI',
    #            solver_options={"max_boson_occ":3},
    #            maxiter=30,
    #            max_elec_err=1e-6
    #    )
    #    emb.site_fragmentation()
    #    frag = emb.add_atomic_fragment([0, 1])
    #    frag.add_tsymmetric_fragments(tvecs=[7, 1, 1])
    #    emb.kernel()

    #    known_values = {'e_tot':-16.63125078900363}

    #    self._test_energy(emb, known_values)

    #FIXME bug #9
    #def test_6x6_u0_1x1imp_2occ(self):
    #    """Tests for 6x6 U=0 Hubbard model with single site impurities.
    #    """

    #    emb = edmet.EDMET(
    #            testsystems.hubb_6x6_u0_1x1imp.rhf(),
    #            solver='FCI',
    #            max_boson_occ=2,
    #    )
    #    emb.site_fragmentation()
    #    frag = emb.add_atomic_fragment([0])
    #    frag.add_tsymmetric_fragments(tvecs=[6, 6, 1])
    #    emb.kernel()

    #    known_values = {'e_tot': -56.0}

    #    self._test_energy(emb, known_values)

    def test_6x6_u6_1x1imp_2occ(self):
        """Tests for 6x6 U=6 Hubbard model with single site impurities.
        """

        emb = edmet.EDMET(
                testsystems.hubb_6x6_u6_1x1imp.rhf(),
                solver='FCI',
                solver_options={"max_boson_occ":2},
                bosonic_interaction="direct",
                oneshot=True,
                make_dd_moments=False,
        )
        with emb.site_fragmentation() as f:
            frag = f.add_atomic_fragment([0])
            frag.add_tsymmetric_fragments(tvecs=[6, 6, 1])
        emb.kernel()

        known_values = {'e_tot':-49.255623407653644}

        self._test_energy(emb, known_values)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
