import unittest

import numpy as np
from scipy.interpolate import lagrange as lagrange_scipy

from numerical_methods.lagrange import lagrange


class TestLagrange(unittest.TestCase):
    def test_lagrange(self):
        x = np.array([0, 1, 2, 3])
        y = np.array([0, 1, 4, 9])
        z = np.array([0.5, 1.5, 2.5])

        np.testing.assert_allclose(
            lagrange(x, y, z),
            lagrange_scipy(x, y)(z),
        )
