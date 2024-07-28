import unittest

import numpy as np
from scipy import integrate

from numerical_methods.trapezoidal import trapezoidal


class TestTrapezoidal(unittest.TestCase):
    def test_trapezoidal(self):
        a, b = 0, 1

        def f(x):
            return x**2

        assert np.isclose(
            integrate.quad(f, a, b)[0],
            trapezoidal(f, a, b, 1000),
        )

    def test_slides_exercise(self):
        f, a, b = np.exp, 0, 1

        assert np.isclose(
            1.71971,  # https://www.wolframalpha.com/input?i=integrate+%5B%2F%2Fmath%3Ae%5Ex%2F%2F%5Ddx+using+trapezoidal+rule+with+%5B%2F%2Fnumber%3A10%2F%2F%5D+intervals+from+x%3D%5B%2F%2Fnumber%3A0%2F%2F%5D+to+%5B%2F%2Fnumber%3A1%2F%2F%5D
            trapezoidal(f, a, b, 10),
        )
