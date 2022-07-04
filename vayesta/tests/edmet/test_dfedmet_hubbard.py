import unittest

from vayesta import edmet
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class HubbardDFEDMETTests(TestCase):
    PLACES_ENERGY = 9

    def _test_energy(self, emb, dfedmet, known_values):
        """Test that the energy contributions match a known value and the non-density fitted value.
        """
        self.assertAlmostEqual(dfedmet.e_tot - dfedmet.e_nonlocal, known_values['e_clus'], self.PLACES_ENERGY)
        self.assertAlmostEqual(dfedmet.e_nonlocal, known_values['e_nl'], self.PLACES_ENERGY)
        self.assertAlmostEqual(emb.e_tot, dfedmet.e_tot, self.PLACES_ENERGY)

    def test_14_upoint4_2imp_4occ(self):
        """Tests for N=14 U=4 Hubbard model with double site impurities and density fitting, single shot.
        Self-consistent (E)DMET is pretty unstable on the Hubbard model, so further tests will have to wait for UHF
        (not long!).
        """
        emb = edmet.EDMET(
            testsystems.hubb_14_u4.rhf(),
            solver='EBFCI',
            solver_options={"max_boson_occ":2},
            maxiter=1,
            max_elec_err=1e-6
        )
        with emb.site_fragmentation() as f:
            frag = f.add_atomic_fragment([0, 1])
            frag.add_tsymmetric_fragments(tvecs=[7, 1, 1])
        emb.kernel()

        dfedmet = edmet.EDMET(
            testsystems.hubb_14_u4_df.rhf(),
                solver='EBFCI',
                solver_options={"max_boson_occ":2},
                maxiter=1,
                max_elec_err=1e-6
        )
        with dfedmet.site_fragmentation() as f:
            frag = f.add_atomic_fragment([0, 1])
            frag.add_tsymmetric_fragments(tvecs=[7, 1, 1])
        dfedmet.kernel()
        # This is the energy without the nonlocal correlation energy RPA correction.
        known_values = {'e_clus': -8.01015928145061, 'e_nl': -0.5338487284590787}

        self._test_energy(emb, dfedmet, known_values)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
