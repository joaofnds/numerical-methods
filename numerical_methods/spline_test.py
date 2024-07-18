import unittest

import numpy as np

from numerical_methods.spline import linear_spline


class TestSpline(unittest.TestCase):
    def test_spline(self):
        start, stop = 0, 2 * np.pi
        f = np.sin
        x = np.linspace(start, stop, 10)
        xvals = np.linspace(start, stop, 100)

        assert np.allclose(
            f(xvals),
            linear_spline(x, f(x), xvals),
            rtol=0.1,
        )
