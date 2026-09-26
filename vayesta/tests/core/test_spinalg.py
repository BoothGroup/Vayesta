import pytest
import numpy as np
import scipy.linalg

from vayesta.core import spinalg
from vayesta.tests.common import TestCase


def _random_hermitian(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.random((n, n))
    return a + a.T


@pytest.mark.fast
class TestEigh(TestCase):
    def test_restricted(self):
        a = _random_hermitian(5, 0)
        e, v = spinalg.eigh(a)
        e_ref, v_ref = scipy.linalg.eigh(a)
        self.assertAllclose(e, e_ref)
        self.assertAllclose(np.abs(v), np.abs(v_ref))

    def test_unrestricted(self):
        a = (_random_hermitian(5, 0), _random_hermitian(5, 1))
        e, v = spinalg.eigh(a)
        for s in range(2):
            e_ref, v_ref = scipy.linalg.eigh(a[s])
            self.assertAllclose(e[s], e_ref)
            self.assertAllclose(np.abs(v[s]), np.abs(v_ref))

    def test_unrestricted_generalized(self):
        a = (_random_hermitian(4, 0), _random_hermitian(4, 1))
        b = np.eye(4) + 0.1 * _random_hermitian(4, 2)
        e, v = spinalg.eigh(a, b)
        for s in range(2):
            e_ref, _ = scipy.linalg.eigh(a[s], b=b)
            self.assertAllclose(e[s], e_ref)
