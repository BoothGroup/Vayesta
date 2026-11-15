import unittest
from unittest import mock
import logging
from types import SimpleNamespace

import numpy as np

import vayesta.ewf
from vayesta import rpa
from vayesta.core.bosonic_bath.qba_rpa import RPA_QBA_Bath
from vayesta.core.bosonic_bath.bbath import Boson_Threshold
from vayesta.core.qemb.qemb import Embedding
from vayesta.core.util import dot
from vayesta.rpa.rirpa import RIdRRPA
from vayesta.rpa.rirpa.RIdRRPA import ssRIdRRPA
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase

THRESHOLD = 1e-4
TIGHT = dict(eta0_tol=1e-12)
CLENCUR = dict(eta0_method="clencur")
ALL_ON = dict(eta0_method="hht", singlet_target=True, singlet_projection=True, **TIGHT)


class BathModeTest(TestCase):
    def test_singlet_default_eligibility(self) -> None:
        """Automatic singlet baths require a real, paired, restricted spin-zero reference."""
        cases = (
            ("restricted", 0, [2.0, 0.0], np.eye(2), True),
            ("unrestricted", 0, [2.0, 0.0], np.eye(2), False),
            ("restricted", 2, [2.0, 0.0], np.eye(2), False),
            ("restricted", 0, [1.0, 1.0], np.eye(2), False),
            ("restricted", 0, [1.5, 0.5], np.eye(2), False),
            ("restricted", 0, [2.0, 0.0], np.eye(2, dtype=complex), False),
        )
        for spinsym, spin, occupations, coeff, expected in cases:
            with self.subTest(spinsym=spinsym, spin=spin, occupations=occupations, dtype=coeff.dtype):
                emb = object.__new__(Embedding)
                emb.spinsym = spinsym
                emb.mf = SimpleNamespace(mol=SimpleNamespace(spin=spin), mo_occ=np.array(occupations), mo_coeff=coeff)
                self.assertIs(emb._default_singlet_bath, expected)

    def test_weak_positive_mode_retained(self) -> None:
        """Retain a finite positive mode above the requested occupation cutoff."""
        co = np.ones((1, 1))
        cv = np.array([[1.0, 0.0, 0.0]])

        def get_overlap(key: str) -> np.ndarray:
            """Return the occupied or virtual cluster overlap."""
            return co if "occ" in key else cv

        fragment = SimpleNamespace(log=logging.getLogger(__name__), get_overlap=get_overlap)
        spatial = np.array([[0.0, np.sqrt(0.1 / 2), 0.0], [0.0, 0.0, np.sqrt(1e-14 / 2)]])
        for singlet_projection in (False, True):
            with self.subTest(singlet_projection=singlet_projection):
                bath = object.__new__(RPA_QBA_Bath)
                bath.fragment = fragment
                bath.target_m0 = np.hstack((spatial, spatial))
                bath.singlet_projection = singlet_projection
                bath.coeff, bath.occup = bath.make_boson_coeff()
                coeff, _ = bath.get_bath(boson_threshold=Boson_Threshold("occupation", 1e-15), verbose=False)
                self.assertEqual(coeff.shape, (2, 6))
                self.assertAllclose(bath.occup, [0.1, 1e-14], atol=0, rtol=1e-12)
                self.assertAllclose(coeff @ coeff.T, np.eye(2), atol=1e-12, rtol=0)


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

    def _baths(
        self, target_orbitals, local_projection, exact=False, threshold=THRESHOLD, fragment_options=None, **options
    ):
        emb = vayesta.ewf.EWF(
            self.mf,
            bath_options=dict(bathtype="mp2", threshold=1e-4),
            bosonic_bath_options=dict(
                bathtype="rpa",
                target_orbitals=target_orbitals,
                local_projection=local_projection,
                threshold=threshold,
                **options,
            ),
        )
        with emb.iao_fragmentation() as f:
            f.add_all_atomic_fragments()
        if fragment_options is not None:
            emb.fragments[0].opts.bosonic_bath_options.update(fragment_options)
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
                for options in (
                    CLENCUR,
                    TIGHT,
                    dict(singlet_target=False, singlet_projection=False, **TIGHT),
                    dict(singlet_target=False, singlet_projection=True, **TIGHT),
                    dict(singlet_target=True, singlet_projection=False, **TIGHT),
                ):
                    with self.subTest(target=target_orbitals, projection=local_projection, **options):
                        self._assert_same_bath(self._baths(target_orbitals, local_projection, **options), refs)
                with self.subTest(target=target_orbitals, projection=local_projection, defaults=True):
                    # The bath's default eta0_tol, min(1e-6, THRESHOLD / 100) = 1e-6, gives errors of about 5e-5.
                    self._assert_same_bath(self._baths(target_orbitals, local_projection), refs, tol=1e-4)

    def test_paths_used(self):
        """Automatic defaults and all explicit overrides select the requested paths independently."""
        singlet_bath = RPA_QBA_Bath._make_boson_coeff_singlet
        results = []

        def spy(bath):
            results.append(singlet_bath(bath))
            return results[-1]

        cases = (
            ({}, True, True, True),
            (ALL_ON, True, True, True),
            (dict(singlet_target=False, singlet_projection=False), True, False, False),
            (dict(singlet_target=False, singlet_projection=True), True, False, True),
            (dict(singlet_target=True, singlet_projection=False), True, True, False),
            (CLENCUR, False, True, True),
        )
        for options, expect_hht, expect_target, expect_projection in cases:
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
            with self.subTest(**options):
                self.assertEqual(hht.call_count, int(expect_hht))
                self.assertEqual(singlet_target.call_count, int(expect_target))
                self.assertEqual(len(results), len(baths) if expect_projection else 0)
                self.assertTrue(all(r is not None for r in results))
                for bath in baths:
                    ov = bath.coeff.shape[1] // 2
                    self.assertTrue(np.all(np.isfinite(bath.coeff)))
                    if expect_projection:
                        self.assertTrue(np.array_equal(bath.coeff[:, :ov], bath.coeff[:, ov:]))

    def test_ineligible_reference_defaults(self) -> None:
        """Disable automatic paths for ineligible references while preserving explicit requests."""
        for options, enabled in (({}, False), (ALL_ON, True)):
            with self.subTest(enabled=enabled):
                with (
                    mock.patch.object(
                        Embedding, "_default_singlet_bath", new_callable=mock.PropertyMock, return_value=False
                    ),
                    mock.patch.object(
                        ssRIdRRPA,
                        "_kernel_mom0_singlet_target",
                        autospec=True,
                        side_effect=ssRIdRRPA._kernel_mom0_singlet_target,
                    ) as target,
                    mock.patch.object(
                        RPA_QBA_Bath,
                        "_make_boson_coeff_singlet",
                        autospec=True,
                        side_effect=RPA_QBA_Bath._make_boson_coeff_singlet,
                    ) as projection,
                ):
                    baths = self._baths("dmet", None, **options)
                self.assertEqual(target.call_count, int(enabled))
                self.assertEqual(projection.call_count, len(baths) if enabled else 0)

    def test_fragment_projection_override(self) -> None:
        """Fragment projection choices override both automatic and explicit embedding defaults."""
        for options, fragment_value, other_value in (({}, False, True), (dict(singlet_projection=False), True, False)):
            with self.subTest(fragment_value=fragment_value):
                baths = self._baths("dmet", None, fragment_options=dict(singlet_projection=fragment_value), **options)
                self.assertIs(baths[0].singlet_projection, fragment_value)
                self.assertTrue(all(b.singlet_projection is other_value for b in baths[1:]))

    def test_class_defaults_used(self):
        """The moment method still inherits the RPA class default unless explicitly overridden."""
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

    def test_zero_threshold(self) -> None:
        """A zero bath cutoff keeps a positive HHT tolerance and finite bath coefficients."""
        with mock.patch.object(RIdRRPA, "hht_mom0_spinindependent", wraps=RIdRRPA.hht_mom0_spinindependent) as hht:
            baths = self._baths("dmet", None, threshold=0.0)
        self.assertEqual(hht.call_count, 1)
        self.assertEqual(hht.call_args.kwargs["tol"], ssRIdRRPA.eta0_tol)
        for bath in baths:
            self.assertTrue(np.all(np.isfinite(bath.coeff)))
            self.assertTrue(np.all(bath.occup > 0))
            self.assertAllclose(bath.coeff @ bath.coeff.T, np.eye(len(bath.occup)), atol=1e-10, rtol=0)

    def test_fragment_tolerance(self) -> None:
        """Use the tightest fragment accuracy, including explicit fragment overrides."""
        for options, expected in ((dict(threshold=1e-8), 1e-10), (dict(eta0_tol=1e-12), 1e-12)):
            with self.subTest(**options):
                with mock.patch.object(
                    RIdRRPA, "hht_mom0_spinindependent", wraps=RIdRRPA.hht_mom0_spinindependent
                ) as hht:
                    self._baths("dmet", None, fragment_options=options)
                self.assertEqual(hht.call_count, 1)
                self.assertEqual(hht.call_args.kwargs["tol"], expected)

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
