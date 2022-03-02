import unittest

from vayesta import edmet
from vayesta.tests.cache import latts


class HubbardDFEDMETTests(unittest.TestCase):
    PLACES_ENERGY = 9

    def _test_energy(self, emb, dfedmet, known_values):
        """Test that the energy matches a known value and the non-density fitted value.
        """

        self.assertAlmostEqual(dfedmet.e_tot, known_values['e_tot'], self.PLACES_ENERGY)
        self.assertAlmostEqual(emb.e_tot, dfedmet.e_tot, self.PLACES_ENERGY)

    #FIXME bug #9
    #def test_6_u0_1imp_1occ(self):
    #    """Tests for N=6 U=0 Hubbard model with single site impurities.
    #    """

    #    emb = edmet.EDMET(
    #            latts['hubb_6_u0']['rhf'],
    #            solver='EBFCI',
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
    #            latts['hubb_6_u0']['rhf'],
    #            solver='EBFCI',
    #            max_boson_occ=6,
    #    )
    #    emb.site_fragmentation()
    #    frag = emb.add_atomic_fragment([0, 1])
    #    frag.add_tsymmetric_fragments(tvecs=[3, 1, 1])
    #    emb.kernel()

    #    known_values = {'e_tot': -8.0}

    #    self._test_energy(emb, known_values)

    def test_14_upoint4_2imp_4occ(self):
        """Tests for N=14 U=4 Hubbard model with double site impurities and density fitting, single shot.
        Self-consistent (E)DMET is pretty unstable on the Hubbard model, so further tests will have to wait for UHF
        (not long!).
        """
        emb = edmet.EDMET(
            latts['hubb_14_u4']['rhf'],
            solver='EBFCI',
            max_boson_occ=2,
            maxiter=1,
            max_elec_err=1e-6
        )
        emb.site_fragmentation()
        frag = emb.add_atomic_fragment([0, 1])
        frag.add_tsymmetric_fragments(tvecs=[7, 1, 1])
        emb.kernel()

        dfedmet = edmet.EDMET(
                latts['hubb_14_u4_df']['rhf'],
                solver='EBFCI',
                max_boson_occ=2,
                maxiter=1,
                max_elec_err=1e-6
        )
        dfedmet.site_fragmentation()
        frag = dfedmet.add_atomic_fragment([0, 1])
        frag.add_tsymmetric_fragments(tvecs=[7, 1, 1])
        dfedmet.kernel()

        known_values = {'e_tot': -8.01015928145061}

        self._test_energy(emb, dfedmet, known_values)

    #FIXME bug #9
    #def test_6x6_u0_1x1imp_2occ(self):
    #    """Tests for 6x6 U=0 Hubbard model with single site impurities.
    #    """

    #    emb = edmet.EDMET(
    #            latts['hubb_6x6_u0_1x1imp']['rhf'],
    #            solver='EBFCI',
    #            max_boson_occ=2,
    #    )
    #    emb.site_fragmentation()
    #    frag = emb.add_atomic_fragment([0])
    #    frag.add_tsymmetric_fragments(tvecs=[6, 6, 1])
    #    emb.kernel()

    #    known_values = {'e_tot': -56.0}

    #    self._test_energy(emb, known_values)



if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
