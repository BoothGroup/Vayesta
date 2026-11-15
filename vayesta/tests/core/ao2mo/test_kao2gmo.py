import pytest
import unittest
import itertools
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.df
import pyscf.pbc.tools
from pyscf.pbc.lib.kpts_helper import group_by_conj_pairs
from threadpoolctl import ThreadpoolController

from vayesta.core.ao2mo import kao2gmo_cderi
from vayesta.libs import libcore
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems

kao2gmo = import_module("vayesta.core.ao2mo.kao2gmo")


class AuxiliaryFourierTests(TestCase):
    def test_tiled_fourier_matches_dense_and_checks_every_tile(self) -> None:
        """Check tile boundaries, real storage, and the largest imaginary value."""
        rng = np.random.default_rng(17)
        phase = rng.normal(size=(5, 3)) + 1j * rng.normal(size=(5, 3))
        factors = rng.normal(size=(3, 3, 4, 7)) + 1j * rng.normal(size=(3, 3, 4, 7))
        factors[..., -1, -1] *= 100
        reference = np.tensordot(phase, factors, axes=1)
        with patch.object(kao2gmo, "_FOURIER_TILE_BYTES", 5 * 16 * 5):
            result, im = kao2gmo._auxiliary_fourier_real(phase, factors)
        self.assertAllclose(result, reference.real, atol=1e-12)
        self.assertAlmostEqual(im, np.max(np.abs(reference.imag)), places=11)
        self.assertTrue(result.flags.c_contiguous)
        self.assertFalse(np.iscomplexobj(result.base))

    def test_tiled_fourier_scopes_blas_limit(self) -> None:
        """Check that the tile loop limits BLAS and restores the caller's setting."""
        blas = ThreadpoolController().select(user_api="blas")
        if not blas.info():
            self.skipTest("Requires a controllable BLAS thread pool")
        phase = np.eye(3, dtype=complex)
        factors = np.ones((3, 2, 3, 4), dtype=complex)
        matmul = np.matmul
        tile_thread_counts = []

        def observe_matmul(*args: object, **kwargs: object) -> np.ndarray:
            """Record the active BLAS thread counts at each matrix product."""
            tile_thread_counts.append(tuple(pool["num_threads"] for pool in blas.info()))
            return matmul(*args, **kwargs)

        with blas.limit(limits=2):
            before = blas.info()
            if not any((pool["num_threads"] or 0) > 1 for pool in before):
                self.skipTest("Requires a BLAS pool that can use more than one thread")
            with (
                patch.object(kao2gmo, "_FOURIER_TILE_BYTES", 3 * 16 * 5),
                patch.object(kao2gmo, "_FOURIER_SMALL_MESH_BLAS_THREADS", 1),
                patch.object(np, "matmul", observe_matmul),
            ):
                kao2gmo._auxiliary_fourier_real(phase, factors)
            self.assertGreater(len(tile_thread_counts), 1)
            self.assertTrue(all((count or 0) <= 1 for counts in tile_thread_counts for count in counts))
            self.assertEqual(blas.info(), before)


class ReverseKPairTests(TestCase):
    def test_reuse_matches_full_loop_with_multiple_auxiliary_blocks(self) -> None:
        """Check reuse and momentum routing with distinct forward/reverse sectors."""
        gdf = testsystems.he_k321.rhf().with_df
        nk = len(gdf.kpts)
        coeff = np.random.default_rng(29).standard_normal((nk * gdf.cell.nao, 3))
        reference, _ = kao2gmo_cderi(gdf, coeff, driver="python", tril_kij=False, blksize=2)
        result, negative = kao2gmo_cderi(gdf, coeff, driver="python", blksize=2)
        self.assertIsNone(negative)
        self.assertAllclose(result, reference)

        if libcore is None:
            self.skipTest("Requires the Vayesta native library to check the kernel calls")
        with patch.object(libcore, "ao2mo_cderi", wraps=libcore.ao2mo_cderi) as transform:
            result, _ = kao2gmo_cderi(gdf, (coeff, coeff.copy()), driver="c", blksize=2)
        nblocks = (gdf.auxcell.nao_nr() + 1) // 2
        self.assertEqual(transform.call_count, nblocks * nk * (nk + 1) // 2)
        self.assertAllclose(result, reference)

    def test_distinct_orbital_spaces_use_both_contractions(self) -> None:
        """Equal dimensions do not imply equal orbital coefficient matrices."""
        if libcore is None:
            self.skipTest("Requires the Vayesta native library to check the kernel calls")
        gdf = testsystems.he_k321.rhf().with_df
        nk = len(gdf.kpts)
        rng = np.random.default_rng(39)
        coeff = tuple(rng.standard_normal((nk * gdf.cell.nao, 3)) for _ in range(2))
        reference, _ = kao2gmo_cderi(gdf, coeff, driver="python", tril_kij=False)
        with patch.object(libcore, "ao2mo_cderi", wraps=libcore.ao2mo_cderi) as transform:
            result, _ = kao2gmo_cderi(gdf, coeff, driver="c")
        self.assertEqual(transform.call_count, nk**2)
        self.assertAllclose(result, reference)

    def test_kpoints_without_partners_use_reverse_contraction(self) -> None:
        """Reuse only pairs whose k-points both have their conjugate partners."""
        if libcore is None:
            self.skipTest("Requires the Vayesta native library to check the kernel calls")
        gdf = testsystems.he_k321.rhf().with_df
        full_kpts = gdf.kpts
        pairs = group_by_conj_pairs(gdf.cell, full_kpts, return_kpts_pairs=False)
        missing = next(j for i, j in pairs if j is not None and i != j)
        keep = np.arange(len(full_kpts)) != missing
        kpts = full_kpts[keep]
        nk = len(kpts)
        coeff = np.random.default_rng(49).standard_normal((nk * gdf.cell.nao, 3))
        subset = SimpleNamespace(cell=gdf.cell, auxcell=gdf.auxcell, kpts=kpts, sr_loop=gdf.sr_loop)
        scell, phase = pyscf.pbc.tools.k2gamma.get_phase(gdf.cell, full_kpts)
        # Isolate the k-pair loop on an incomplete grid. Use the matching
        # submatrix of phases and compare the k-space factors before unfolding.
        with patch.object(pyscf.pbc.tools.k2gamma, "get_phase", return_value=(scell, phase[:nk, keep])):
            reference, _ = kao2gmo_cderi(subset, coeff, driver="python", tril_kij=False, make_real=False)
            with patch.object(libcore, "ao2mo_cderi", wraps=libcore.ao2mo_cderi) as transform:
                result, _ = kao2gmo_cderi(subset, coeff, driver="c", make_real=False)
        paired = nk - 1
        self.assertEqual(transform.call_count, nk**2 - paired * (paired - 1) // 2)
        self.assertAllclose(result, reference)


class NativeDFThreadingTests(TestCase):
    def test_native_cderi_limits_blas_only_during_parallel_contractions(self) -> None:
        """Check the effective thread limits and numerical result of the native path."""
        if libcore is None or not hasattr(libcore, "ao2mo_cderi_omp_threads"):
            self.skipTest("Requires the native Vayesta library with the OpenMP query")
        controller = ThreadpoolController()
        blas = controller.select(internal_api="openblas").select(threading_layer="pthreads")
        if not blas.info():
            self.skipTest("Requires a controllable pthread OpenBLAS pool")

        gdf = testsystems.he_k321.rhf().with_df
        coeff = np.random.default_rng(738).standard_normal((len(gdf.kpts) * gdf.cell.nao, 3))
        reference, _ = kao2gmo_cderi(gdf, coeff, driver="python")
        native = libcore.ao2mo_cderi
        fourier = kao2gmo._auxiliary_fourier_real
        omp_threads = libcore.ao2mo_cderi_omp_threads
        native_counts = []
        fourier_counts = []

        def blas_counts() -> tuple[int, ...]:
            """Read the thread counts of the selected BLAS pools."""
            return tuple(pool["num_threads"] for pool in blas.info())

        def observe_native(*args: object) -> int:
            """Record the BLAS and OpenMP thread counts at kernel entry."""
            native_counts.append((omp_threads(), blas_counts()))
            return native(*args)

        def observe_fourier(*args: object, **kwargs: object) -> tuple[np.ndarray, float]:
            """Record the restored BLAS limits at the Fourier transform."""
            fourier_counts.append(blas_counts())
            return fourier(*args, **kwargs)

        with blas.limit(limits=2):
            caller_counts = blas_counts()
            if not any(count > 1 for count in caller_counts):
                self.skipTest("Requires a BLAS pool that can use more than one thread")
            with controller.limit(limits=2, user_api="openmp"):
                if omp_threads() != 2:
                    self.skipTest("Requires a controllable Vayesta OpenMP team")
            for outer_threads in (1, 2):
                with self.subTest(outer_threads=outer_threads):
                    native_counts.clear()
                    fourier_counts.clear()
                    with (
                        controller.limit(limits=outer_threads, user_api="openmp"),
                        patch.object(libcore, "ao2mo_cderi", observe_native),
                        patch.object(kao2gmo, "_auxiliary_fourier_real", observe_fourier),
                    ):
                        result, negative = kao2gmo_cderi(gdf, coeff, driver="c")
                    inner_counts = caller_counts if outer_threads == 1 else (1,) * len(caller_counts)
                    self.assertEqual(set(native_counts), {(outer_threads, inner_counts)})
                    self.assertEqual(set(fourier_counts), {caller_counts})
                    self.assertEqual(blas_counts(), caller_counts)
                    self.assertIsNone(negative)
                    self.assertAllclose(result, reference)


@pytest.mark.slow
class KAO2GMO_Tests(TestCase):
    def test_kao2gmo_cderi_3d(self):
        cell = testsystems.he_k321.mol
        kmf = testsystems.he_k321.rhf()
        kmesh = (3, 2, 1)
        kpts = testsystems.he_k321.kpts
        gdf = kmf.with_df

        nmo1, nmo2 = 1, 2
        np.random.seed(1491)
        mo1 = np.random.rand(len(kpts) * cell.nao, nmo1)
        mo2 = np.random.rand(len(kpts) * cell.nao, nmo2)

        scell = pyscf.pbc.tools.super_cell(cell, kmesh)
        gdf_sc = pyscf.pbc.df.GDF(scell)
        gdf_sc.auxbasis = "def2-svp-ri"
        gdf_sc.build()
        eri_expected = gdf_sc.ao2mo((mo1, mo2, mo1, mo2)).reshape(nmo1, nmo2, nmo1, nmo2)

        options = {
            "driver": ["c", "python"],
            "make_real": [True, False],
            "blksize": [None, 10],
            "tril_kij": [True, False],
        }
        options_comb = [dict(zip(options, val)) for val in itertools.product(*options.values())]

        for opts in options_comb:
            cderi, cderi_neg = kao2gmo_cderi(gdf, (mo1, mo2), **opts)
            eri = np.einsum("Lij,Lkl->ijkl", cderi.conj(), cderi)
            self.assertIsNone(cderi_neg)
            self.assertIsNone(np.testing.assert_almost_equal(eri, eri_expected))

    def test_kao2gmo_cderi_2d(self):
        cell = testsystems.he_k32.mol
        kmf = testsystems.he_k32.rhf()
        kmesh = (3, 2, 1)
        kpts = testsystems.he_k32.kpts
        gdf = kmf.with_df

        nmo1, nmo2 = 1, 2
        np.random.seed(15918)
        mo1 = np.random.rand(len(kpts) * cell.nao, nmo1)
        mo2 = np.random.rand(len(kpts) * cell.nao, nmo2)

        scell = pyscf.pbc.tools.super_cell(cell, kmesh)
        gdf_sc = pyscf.pbc.df.GDF(scell)
        gdf_sc.auxbasis = "def2-svp-ri"
        gdf_sc.build()
        eri_expected = gdf_sc.ao2mo((mo1, mo2, mo1, mo2)).reshape(nmo1, nmo2, nmo1, nmo2)

        options = {
            "driver": ["c", "python"],
            "make_real": [True, False],
            "blksize": [None, 10],
            "tril_kij": [True, False],
        }
        options_comb = [dict(zip(options, val)) for val in itertools.product(*options.values())]

        for opts in options_comb:
            cderi, cderi_neg = kao2gmo_cderi(gdf, (mo1, mo2), **opts)
            eri = np.einsum("Lij,Lkl->ijkl", cderi.conj(), cderi) - np.einsum(
                "Lij,Lkl->ijkl", cderi_neg.conj(), cderi_neg
            )
            self.assertIsNone(np.testing.assert_almost_equal(eri, eri_expected))

    def test_kao2gmo_cderi_1d(self):
        cell = testsystems.he_k3.mol
        kmf = testsystems.he_k3.rhf()
        kmesh = (3, 1, 1)
        kpts = testsystems.he_k3.kpts
        gdf = kmf.with_df

        nmo1, nmo2 = 1, 2
        np.random.seed(10518)
        mo1 = np.random.rand(len(kpts) * cell.nao, nmo1)
        mo2 = np.random.rand(len(kpts) * cell.nao, nmo2)

        scell = pyscf.pbc.tools.super_cell(cell, kmesh)
        gdf_sc = pyscf.pbc.df.GDF(scell)
        gdf_sc.auxbasis = "def2-svp-ri"
        gdf_sc.build()
        eri_expected = gdf_sc.ao2mo((mo1, mo2, mo1, mo2)).reshape(nmo1, nmo2, nmo1, nmo2)

        options = {
            "driver": ["c", "python"],
            "make_real": [True, False],
            "blksize": [None, 10],
            "tril_kij": [True, False],
        }
        options_comb = [dict(zip(options, val)) for val in itertools.product(*options.values())]

        for opts in options_comb:
            cderi, cderi_neg = kao2gmo_cderi(gdf, (mo1, mo2), **opts)
            eri = np.einsum("Lij,Lkl->ijkl", cderi.conj(), cderi) - np.einsum(
                "Lij,Lkl->ijkl", cderi_neg.conj(), cderi_neg
            )
            self.assertIsNone(np.testing.assert_almost_equal(eri, eri_expected))


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
