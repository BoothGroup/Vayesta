import unittest
from unittest import mock

import numpy as np

import vayesta.ewf
from vayesta import rpa
from vayesta.core.bosonic_bath.qba_rpa import RPA_QBA_Bath
from vayesta.core.util import dot
from vayesta.rpa.rirpa import RIdRRPA
from vayesta.rpa.rirpa.RIdRRPA import ssRIdRRPA
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase

THRESHOLD = 1e-4
TIGHT = dict(eta0_tol=1e-12)
CLENCUR = dict(eta0_method="clencur")
ALL_ON = dict(eta0_method="hht", singlet_target=True, singlet_projection=True, **TIGHT)


class RPABosonicBathHHTTest(TestCase):
    """RPA bosonic baths with the HHT moment, singlet target and singlet projection, against exact RPA moments.

    Only the bath is built (no solver), for every target space and local projection.
    """

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_ccpvdz_df.rhf()
        exact = rpa.ssRPA(cls.mf)
        exact.kernel()
        cls.m0_exact = exact.gen_moms(0)[0]

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.m0_exact

    def _baths(self, target_orbitals, local_projection, exact=False, **options):
        emb = vayesta.ewf.EWF(
            self.mf,
            bath_options=dict(bathtype="mp2", threshold=1e-4),
            bosonic_bath_options=dict(
                bathtype="rpa",
                target_orbitals=target_orbitals,
                local_projection=local_projection,
                threshold=THRESHOLD,
                **options,
            ),
        )
        with emb.iao_fragmentation() as f:
            f.add_all_atomic_fragments()
        for x in emb.fragments:
            x.make_bath()
            x.make_cluster()
        if exact:
            moms = lambda rirpa, max_moment, target_rot=None, **kwargs: ((target_rot @ self.m0_exact)[None], None)
            with mock.patch.object(ssRIdRRPA, "kernel_moms", moms):
                emb.build_bosonic_bath()
        else:
            emb.build_bosonic_bath()
        return [x._boson_bath_factory for x in emb.fragments]

    def _assert_same_bath(self, baths, refs, tol=1e-8):
        for bath, ref in zip(baths, refs):
            k = int(np.sum(ref.occup >= THRESHOLD))
            self.assertEqual(int(np.sum(bath.occup >= THRESHOLD)), k)
            if k == 0:
                continue
            self.assertAllclose(bath.occup[:k], ref.occup[:k], rtol=tol, atol=0)
            # Largest sine of the angles between the retained bath subspaces.
            qa = np.linalg.qr(ref.coeff[:k].T)[0]
            qb = np.linalg.qr(bath.coeff[:k].T)[0]
            self.assertLess(np.linalg.norm(qb - qa @ (qa.T @ qb), 2), tol)

    def test_vs_exact(self):
        for target_orbitals in ("full", "dmet"):
            for local_projection in (None, "fragment", "fragment-occ", "fragment-ov"):
                refs = self._baths(target_orbitals, local_projection, exact=True)
                for options in (CLENCUR, TIGHT, ALL_ON, dict(singlet_projection=True, **TIGHT)):
                    with self.subTest(target=target_orbitals, projection=local_projection, **options):
                        self._assert_same_bath(self._baths(target_orbitals, local_projection, **options), refs)
                with self.subTest(target=target_orbitals, projection=local_projection, defaults=True):
                    # The bath's default eta0_tol, min(1e-6, THRESHOLD / 100) = 1e-6, gives errors of about 5e-5.
                    self._assert_same_bath(self._baths(target_orbitals, local_projection), refs, tol=1e-4)

    def test_paths_used(self):
        """With the options on, each new path is taken; the singlet bath also when the target moment is stacked."""
        singlet_bath = RPA_QBA_Bath._make_boson_coeff_singlet
        results = []

        def spy(bath):
            results.append(singlet_bath(bath))
            return results[-1]

        for options in (ALL_ON, dict(singlet_projection=True, **CLENCUR)):
            results.clear()
            with (
                mock.patch.object(RPA_QBA_Bath, "_make_boson_coeff_singlet", spy),
                mock.patch.object(RIdRRPA, "hht_mom0_spinindependent", wraps=RIdRRPA.hht_mom0_spinindependent) as hht,
                mock.patch.object(
                    ssRIdRRPA,
                    "_kernel_mom0_singlet_target",
                    autospec=True,
                    side_effect=ssRIdRRPA._kernel_mom0_singlet_target,
                ) as singlet_target,
            ):
                baths = self._baths("full", "fragment", **options)
            on = options.get("eta0_method") == "hht"
            with self.subTest(**options):
                self.assertEqual(hht.call_count, int(on))
                self.assertEqual(singlet_target.call_count, int(on))
                self.assertEqual(len(results), len(baths))
                self.assertTrue(all(r is not None for r in results))
                for bath in baths:
                    ov = bath.coeff.shape[1] // 2
                    self.assertTrue(np.all(np.isfinite(bath.coeff)))
                    self.assertTrue(np.array_equal(bath.coeff[:, :ov], bath.coeff[:, ov:]))

    def test_class_defaults_used(self):
        """Options left as None keep the class attributes, which external codes may set."""
        with mock.patch.object(ssRIdRRPA, "eta0_method", "clencur"):
            with mock.patch.object(RIdRRPA, "hht_mom0_spinindependent") as hht:
                self._baths("full", "fragment")
        hht.assert_not_called()
        with (
            mock.patch.object(ssRIdRRPA, "eta0_method", "clencur"),
            mock.patch.object(RIdRRPA, "hht_mom0_spinindependent", wraps=RIdRRPA.hht_mom0_spinindependent) as hht,
        ):
            self._baths("full", "fragment", eta0_method="hht")
        self.assertEqual(hht.call_count, 1)

    def test_rank_deficient_target(self):
        """Null modes of a rank-deficient environment Gram matrix are discarded instead of giving NaN coefficients.

        Duplicating the target rows gives a Gram matrix with at least as many round-off eigenvalues as rows in the
        original target, some of them negative; the retained modes are those of the original target with doubled
        weights.
        """
        for singlet_projection in (False, True):
            with self.subTest(singlet_projection=singlet_projection):
                bath = self._baths("full", "fragment", singlet_projection=singlet_projection)[0]
                ref_coeff, ref_occup = bath.coeff, bath.occup
                nrow = bath.target_m0.shape[0]
                bath.target_m0 = np.vstack((bath.target_m0, bath.target_m0))
                if singlet_projection:
                    self.assertIsNotNone(bath._make_boson_coeff_singlet())
                coeff, occup = bath.make_boson_coeff()
                self.assertTrue(np.all(np.isfinite(coeff)))
                self.assertTrue(np.all(occup > 0))
                self.assertLessEqual(len(occup), nrow)
                self.assertEqual(coeff.shape, (len(occup), bath.target_m0.shape[1]))
                self.assertAllclose(dot(coeff, coeff.T), np.eye(len(occup)), atol=1e-8, rtol=0)
                # Retained modes are unchanged.
                k = int(np.sum(ref_occup >= THRESHOLD))
                self.assertGreater(k, 0)
                self.assertAllclose(occup[:k], 2 * ref_occup[:k], rtol=1e-8, atol=0)
                self.assertAllclose(np.abs(dot(coeff[:k], ref_coeff[:k].T)), np.eye(k), atol=1e-6, rtol=0)

    def test_singlet_projection_spin_blocks_differ(self):
        """A target moment with different spin blocks uses the spin-orbital construction."""
        bath = self._baths("full", "fragment")[0]
        m0 = bath.target_m0.copy()
        m0[:, : m0.shape[1] // 2] *= 1.01
        bath.target_m0 = m0
        bath.singlet_projection = True
        self.assertIsNone(bath._make_boson_coeff_singlet())
        occup = bath.make_boson_coeff()[1]
        bath.singlet_projection = False
        self.assertAllclose(occup, bath.make_boson_coeff()[1])


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
