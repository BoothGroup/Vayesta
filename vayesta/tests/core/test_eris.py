import numpy as np

from vayesta.core import eris
from vayesta.core.bosonic_bath.projected_interactions import BosonicHamiltonianProjector
from vayesta.core.util import einsum
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class _Projector(BosonicHamiltonianProjector):
    """Projector with only the attributes required for the loop over the density-fitted integrals."""

    def __init__(self, mf):
        self.mf = mf


class TestLoopCDERI(TestCase):
    """The blocked loops over the three-center integrals have to yield each block exactly once."""

    def _get_loops(self, mf, blksize):
        return {
            "loop_cderi_ao": lambda: eris.loop_cderi_ao(mf, blksize=blksize),
            "BosonicHamiltonianProjector._loop_df": lambda: _Projector(mf)._loop_df(blksize=blksize),
        }

    def _test(self, mf, has_neg):
        nao = mf.mol.nao
        naux = eris.get_naux(mf.with_df)
        eri_ref = mf.with_df.ao2mo(np.eye(nao), compact=False).reshape(4 * [nao])
        for blksize in (None, 1, 3):
            for name, loop in self._get_loops(mf, blksize).items():
                with self.subTest(loop=name, blksize=blksize):
                    cderi = np.zeros((naux, nao, nao))
                    cderi_neg = None
                    nblk = naux_yielded = 0
                    for blk, lab in loop():
                        if blk is None:
                            self.assertIsNone(cderi_neg)
                            cderi_neg = lab
                        else:
                            self.assertEqual(blk.start, naux_yielded)
                            naux_yielded = blk.stop
                            cderi[blk] = lab
                            nblk += 1
                    # For low-dimensional systems, part of the auxiliary space belongs to the negative part:
                    self.assertLessEqual(naux_yielded, naux)
                    if blksize is not None:
                        self.assertEqual(nblk, -(-naux_yielded // blksize))
                    self.assertEqual(cderi_neg is not None, has_neg)
                    eri = einsum("Lab,Lcd->abcd", cderi, cderi)
                    if cderi_neg is not None:
                        eri -= einsum("Lab,Lcd->abcd", cderi_neg, cderi_neg)
                    self.assertAllclose(eri, eri_ref)

        # MO transformation:
        mo_coeff = mf.mo_coeff
        nmo = mo_coeff.shape[-1]
        eri_ref = mf.with_df.ao2mo(mo_coeff, compact=False).reshape(4 * [nmo])
        cderi, cderi_neg = eris.get_cderi_df(mf, mo_coeff)
        self.assertEqual(cderi.shape, (naux, nmo, nmo))
        self.assertEqual(cderi_neg is not None, has_neg)
        eri = einsum("Lij,Lkl->ijkl", cderi, cderi)
        if cderi_neg is not None:
            eri -= einsum("Lij,Lkl->ijkl", cderi_neg, cderi_neg)
        self.assertAllclose(eri, eri_ref)

    def test_molecule(self):
        self._test(testsystems.water_631g_df.rhf(), has_neg=False)

    def test_solid_3d(self):
        self._test(testsystems.h2_sto3g_s311.rhf(), has_neg=False)

    def test_solid_2d(self):
        self._test(testsystems.graphene_sto3g_s211.rhf(), has_neg=True)
