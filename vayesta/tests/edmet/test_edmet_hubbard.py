import unittest

from vayesta import edmet
from vayesta.tests.cache import latts


class EDMET_Hubbard_Tests(unittest.TestCase):
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
    #            latts['hubb_6_u0']['rhf'],
    #            solver='EBFCI',
    #            bos_occ_cutoff=1,
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
    #            bos_occ_cutoff=6,
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
                latts['hubb_10_u2']['rhf'],
                solver='EBFCI',
                bos_occ_cutoff=2,
        )
        emb.site_fragmentation()
        frag = emb.add_atomic_fragment([0, 1])
        frag.add_tsymmetric_fragments(tvecs=[5, 1, 1])
        emb.kernel()

        known_values = {'e_tot': -8.682074457504335}

        self._test_energy(emb, known_values)

    # FIXME 10 site / 2 impurity Hubbard model with 1/8 T-vector?
    #def test_16_u4_2imp_4occ(self):
    #    """Tests for N=10 U=2 Hubbard model with double site impurities.
    #    """

    #    emb = edmet.EDMET(
    #            latts['hubb_10_u2']['rhf'],
    #            solver='EBFCI',
    #            bos_occ_cutoff=4,
    #    )
    #    emb.site_fragmentation()
    #    frag = emb.add_atomic_fragment([0, 1])
    #    #frag.add_tsymmetric_fragments(tvecs=[8, 1, 1])
    #    emb.kernel()

    #    known_values = {'e_tot': -3.472834250801909}

    #    self._test_energy(emb, known_values)

    #FIXME bug #9
    #def test_6x6_u0_1x1imp_2occ(self):
    #    """Tests for 6x6 U=0 Hubbard model with single site impurities.
    #    """

    #    emb = edmet.EDMET(
    #            latts['hubb_6x6_u0_1x1imp']['rhf'],
    #            solver='EBFCI',
    #            bos_occ_cutoff=2,
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
                latts['hubb_6x6_u6_1x1imp']['rhf'],
                solver='EBFCI',
                bos_occ_cutoff=2,
        )
        emb.site_fragmentation()
        frag = emb.add_atomic_fragment([0])
        frag.add_tsymmetric_fragments(tvecs=[6, 6, 1])
        emb.kernel()

        known_values = {'e_tot': -41.29530580982776}

        self._test_energy(emb, known_values)

    def test_8x8_u2_2x1imp_5occ(self):
        """Tests for 8x8 U=2 Hubbard model with 2x1 impurities.
        """

        emb = edmet.EDMET(
                latts['hubb_8x8_u2_2x1imp']['rhf'],
                solver='EBFCI',
                bos_occ_cutoff=5,
        )
        emb.site_fragmentation()
        frag = emb.add_atomic_fragment([0, 1])
        frag.add_tsymmetric_fragments(tvecs=[4, 8, 1])
        emb.kernel()

        known_values = {'e_tot': -84.96249789942502}

        self._test_energy(emb, known_values)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
