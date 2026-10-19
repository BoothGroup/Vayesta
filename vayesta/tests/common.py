import unittest
import numpy as np


class temporary_seed:
    def __init__(self, seed):
        self.seed, self.state = seed, None

    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, *args):
        np.random.set_state(self.state)


class TestCase(unittest.TestCase):
    # Default tolerances of assertAllclose. The absolute tolerance has to cover the convergence thresholds of the
    # solvers (typically 1e-8) and the precision of the stored reference values; a tighter value makes many tests
    # platform dependent. Subclasses can override these attributes.
    allclose_atol = 1e-7
    allclose_rtol = 1e-8

    def assertAllclose(self, actual, desired, rtol=None, atol=None, **kwargs):
        # Resolve the defaults at call time, such that subclasses can override the class attributes:
        if rtol is None:
            rtol = self.allclose_rtol
        if atol is None:
            atol = self.allclose_atol
        # Compare multiple pairs of arrays:
        if isinstance(actual, (tuple, list)):
            for i in range(len(actual)):
                self.assertAllclose(actual[i], desired[i], rtol=rtol, atol=atol, **kwargs)
            return
        # Compare single pair of arrays:
        try:
            np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, **kwargs)
        except AssertionError as e:
            # Add higher precision output:
            message = e.args[0]
            args = e.args[1:]
            message += "\nHigh precision:\n x: %r\n y: %r" % (actual, desired)
            e.args = (message, *args)
            raise


if __name__ == "__main__":
    a = np.random.rand(3, 4)
    b = np.random.rand(4, 5)

    test = TestCase()

    test.assertAllclose(a, a)
    test.assertAllclose((a, b), (a, b))
    test.assertAllclose([[a, a], b], [[a, a], b])
    test.assertAllclose(a, b)
