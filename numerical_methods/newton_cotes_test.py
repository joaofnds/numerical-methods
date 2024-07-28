import unittest

import numpy as np
from scipy import integrate

from numerical_methods.newton_cotes import newton_cotes


class TestNewtonCotes(unittest.TestCase):
    def test_newton_cotes(self):
        a, b = 0, 1

        def f(x):
            return x**2

        assert np.isclose(
            integrate.quad(f, a, b)[0],
            newton_cotes(f, a, b, 1000),
        )
