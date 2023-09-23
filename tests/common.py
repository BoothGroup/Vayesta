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
    allclose_atol = 1e-8
    allclose_rtol = 1e-7

    def assertAllclose(self, actual, desired, rtol=allclose_atol, atol=allclose_rtol, **kwargs):
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
