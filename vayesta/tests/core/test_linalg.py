import unittest
import numpy as np

from vayesta.core import linalg
from vayesta.tests.common import TestCase


class LinalgTests(TestCase):
    def test_recursive_block_svd(self):
        """Test the recursive_block_svd function.
        """

        n = 100
        np.random.seed(1)
        dm = np.random.random((n, n)) - 0.5
        dm += dm.T
        fock = np.random.random((n, n)) - 0.5
        fock += fock.T
        c_frag = np.random.random((n, n//2))
        c_env = np.random.random((n, n-n//2))

        dmocc1 = np.linalg.multi_dot((c_frag.T, fock, c_env))
        u, s, vh = np.linalg.svd(dmocc1)
        mo_svd = np.dot(c_env, vh.T)
        ncpl = len(s)

        dmocc2 = np.linalg.multi_dot((mo_svd[:, :ncpl].T, fock, mo_svd[:, ncpl:]))
        u, s, vh = np.linalg.svd(dmocc2)

        nimp = c_frag.shape[-1]
        c = np.hstack((c_frag, c_env))
        f = np.linalg.multi_dot((c.T, fock, c))
        mo_svd2, sv, orders = linalg.recursive_block_svd(f, n=nimp)
        mo_svd2 = np.dot(c_env, mo_svd2)

        e_svd = np.linalg.eigh(np.dot(mo_svd, mo_svd.T))[0]
        e_svd2 = np.linalg.eigh(np.dot(mo_svd2, mo_svd2.T))[0]
        self.assertAlmostEqual(np.max(np.abs(e_svd-e_svd2)), 0.0, 10)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
