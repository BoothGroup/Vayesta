import unittest
from unittest import mock

import numpy as np

from vayesta import rpa
from vayesta.rpa.rirpa import RIdRRPA
from vayesta.rpa.rirpa.hht import hht_mom0_spinindependent, hht_poles
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


class HHTPolesTest(TestCase):
    def test_tolerance_met(self):
        """The scalar error reported, and the error on a much finer grid, are within the tolerance."""
        for lmin, lmax in [(1.0, 1e2), (0.3, 1e5)]:
            x = np.geomspace(lmin, lmax, 100000)
            for tol in (1e-6, 1e-10):
                shifts, weights, err = hht_poles(lmin, lmax, tol=tol)
                self.assertLessEqual(err, tol)
                self.assertTrue(np.all(shifts > 0) and np.all(weights > 0))
                r = (weights[None] / (x[:, None] + shifts[None])).sum(1)
                self.assertLessEqual(np.max(np.abs(1.0 - np.sqrt(x) * r)), 1.1 * tol)

    def test_fixed_pole_count(self):
        shifts, weights, err = hht_poles(1.0, 1e3, n_poles=7)
        self.assertEqual(len(shifts), 7)
        self.assertEqual(len(weights), 7)
        self.assertLess(err, 1e-3)

    def test_errors(self):
        with self.assertRaises(ValueError):
            hht_poles(0.0, 1.0)
        with self.assertRaises(ValueError):
            hht_poles(2.0, 1.0)
        with self.assertRaises(RuntimeError):
            hht_poles(1.0, 1e6, tol=1e-30)


class HHTMom0Test(TestCase):
    def test_vs_dense(self):
        """0.5 (T eta - T) agrees with the dense eta = D^1/2 M^-1/2 D^1/2, on both solve branches."""
        rng = np.random.default_rng(0)
        nov, naux = 40, 6
        eps = rng.uniform(0.5, 3.0, nov)
        v = 0.3 * rng.standard_normal((naux, nov))
        dh = np.sqrt(eps)
        m = np.diag(eps**2) + 4 * dh[:, None] * (v.T @ v) * dh[None]
        e, u = np.linalg.eigh(m)
        eta = dh[:, None] * ((u * (e**-0.5)[None]) @ u.T) * dh[None]
        for n in (3, 10):  # n < naux: Cholesky solve; n > naux: explicit inverse.
            t = rng.standard_normal((n, nov))
            out, info = hht_mom0_spinindependent(eps, v, t, tol=1e-12)
            self.assertAllclose(out, 0.5 * (t @ eta - t), atol=1e-10, rtol=0)
            self.assertLessEqual(info["scalar_error"], 1e-12)
            self.assertLess(info["max_cholesky_residual"], 1e-12)

    def test_nonpositive_eps(self):
        with self.assertRaises(ValueError):
            hht_mom0_spinindependent(np.array([1.0, 0.0]), np.ones((1, 2)), np.ones((1, 2)))


class MoleculeHHTTest(TestCase):
    """ssRIdRRPA zeroth moment with HHT and the singlet target, against exact RPA and the frequency integral."""

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_ccpvdz_df.rhf()
        exact = rpa.ssRPA(cls.mf)
        exact.kernel()
        cls.moms_exact = exact.gen_moms(2)
        cls.ov = exact.ova
        rng = np.random.default_rng(1)
        cls.t = rng.standard_normal((7, cls.ov))
        cls.t_other = rng.standard_normal((7, cls.ov))

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.moms_exact

    def _rirpa(self, method="hht", singlet=False, **kwargs):
        rirpa = rpa.rirpa.ssRIdRRPA(self.mf, **kwargs)
        rirpa.eta0_method = method
        rirpa.singlet_target = singlet
        return rirpa

    def test_defaults(self):
        """Default behaviour is the frequency integral with spin stacking."""
        self.assertEqual(rpa.rirpa.ssRIdRRPA.eta0_method, "clencur")
        self.assertFalse(rpa.rirpa.ssRIdRRPA.singlet_target)
        self.assertIsNone(rpa.rirpa.ssRIdRRPA.target_blocks)

    def test_moments_vs_exact(self):
        targets = {"same": np.hstack((self.t, self.t)), "different": np.hstack((self.t, self.t_other))}
        for method in ("clencur", "hht"):
            for singlet in (False, True):
                for name, target in targets.items():
                    with self.subTest(method=method, singlet=singlet, target=name):
                        rirpa = self._rirpa(method, singlet)
                        moms, errs = rirpa.kernel_moms(2, target_rot=target)
                        self.assertAllclose(moms, np.einsum("pq,nqr->npr", target, self.moms_exact), atol=1e-9)
                        if method == "hht":
                            self.assertLessEqual(errs[0], rirpa.eta0_tol)
                            self.assertLessEqual(rirpa.hht_info["scalar_error"], rirpa.eta0_tol)

    def test_spatial_vs_exact(self):
        eta = self.moms_exact[0][: self.ov, : self.ov] + self.moms_exact[0][: self.ov, self.ov :]
        for method in ("clencur", "hht"):
            with self.subTest(method=method):
                mom0 = self._rirpa(method).kernel_moms(0, target_rot=self.t, return_spatial=True)[0][0]
                self.assertAllclose(mom0, self.t @ eta, atol=1e-9)

    def test_full_target_vs_exact(self):
        mom0 = self._rirpa("hht", True).kernel_moms(0)[0][0]
        self.assertAllclose(mom0, self.moms_exact[0], atol=1e-9)

    def test_singlet_target_blocks(self):
        """Per-block row bases, including a rank-deficient and an empty block, and inconsistent block counts."""
        rank2 = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [2.0, -1.0]]) @ self.t[:2]
        spatial = np.vstack((rank2, self.t[2:7]))
        target = np.hstack((spatial, spatial))
        ref = target @ self.moms_exact[0]
        for blocks in (None, [4, 0, 5], [9], [3, 3]):
            for method in ("clencur", "hht"):
                with self.subTest(blocks=blocks, method=method):
                    rirpa = self._rirpa(method, True)
                    rirpa.target_blocks = blocks
                    with mock.patch.object(rirpa, "_row_basis", wraps=rirpa._row_basis) as row_basis:
                        mom0 = rirpa.kernel_moms(0, target_rot=target)[0][0]
                    self.assertAllclose(mom0, ref, atol=1e-9)
                    self.assertEqual(row_basis.call_count, 2 if blocks == [4, 0, 5] else 1)

    def test_singlet_target_not_used(self):
        """The singlet path is skipped for different spin blocks and when off."""
        for singlet, target in [
            (True, np.hstack((self.t, self.t_other))),
            (False, np.hstack((self.t, self.t))),
        ]:
            rirpa = self._rirpa("hht", singlet)
            with mock.patch.object(rirpa, "_kernel_mom0_singlet_target") as singlet_path:
                rirpa.kernel_moms(0, target_rot=target)
            singlet_path.assert_not_called()

    def _assert_falls_back(self, rirpa, target, ref=None):
        with mock.patch.object(
            RIdRRPA, "MomzeroDeductHigherOrder_dRHF", wraps=RIdRRPA.MomzeroDeductHigherOrder_dRHF
        ) as ni:
            mom0, errs = rirpa.kernel_moms(0, target_rot=target)
        mom0 = mom0[0]
        ni.assert_called()
        self.assertFalse(hasattr(rirpa, "hht_info"))
        self.assertEqual(errs, (None, None))
        if ref is not None:
            self.assertAllclose(mom0, ref, atol=1e-9)
        return mom0

    def test_fallback_compressed(self):
        """HHT declines A+B factors that are not equal and real (compress > 3); the integral is then used."""
        rirpa = self._rirpa("hht")
        ri = rirpa.get_compressed_MP()[1][0]
        for ri_apb in ([ri, -ri], [ri.astype(complex), ri.astype(complex)]):
            self.assertEqual(rirpa._hht_mom0_spinindependent(self.t, ri_apb), (None, (None, None)))

    def test_fallback_unreachable_tol(self):
        target = np.hstack((self.t, self.t_other))
        rirpa = self._rirpa("hht")
        rirpa.eta0_tol = 1e-30
        self._assert_falls_back(rirpa, target, target @ self.moms_exact[0])

    def test_fallback_negative_cderi(self):
        """With negative CDERIs (as for 2D periodic systems) HHT gives the frequency integral's result."""
        target = np.hstack((self.t, self.t_other))
        lov = self._rirpa().get_cderi()[0].reshape(-1, self.ov)
        neg = 0.01 * np.random.default_rng(2).standard_normal((2, self.ov))
        results = []
        for method in ("clencur", "hht"):
            rirpa = self._rirpa(method)
            rirpa.get_cderi = lambda *args, **kwargs: (lov, neg)
            results.append(self._assert_falls_back(rirpa, target))
        self.assertAllclose(results[1], results[0], atol=1e-12, rtol=0)

    def test_return_niworker(self):
        niworker, offset = self._rirpa("hht")._kernel_mom0(np.hstack((self.t, self.t)), return_niworker=True)
        self.assertIsInstance(niworker, RIdRRPA.MomzeroDeductHigherOrder_dRHF)
        self.assertIsInstance(offset, RIdRRPA.MomzeroOffsetCalcGaussLag)

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            self._rirpa("bogus").kernel_moms(0, target_rot=self.t, return_spatial=True)


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
