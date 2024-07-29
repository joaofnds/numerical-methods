import unittest

import numpy as np

from numerical_methods.runge_kutta import runge_kutta_2, runge_kutta_4


class TestRungeKutta(unittest.TestCase):
    def test_runge_kutta_2(self):
        a, b, y0, n = 0, 1, 1, 10

        def f(x, y):
            return -x * y

        assert np.isclose(
            runge_kutta_2(f, a, b, y0, n)[-1],
            0.60598736,
        )

    def test_runge_kutta_4(self):
        a, b, y0, n = 0, 1, 1, 10

        def f(x, y):
            return -x * y

        assert np.isclose(
            runge_kutta_4(f, a, b, y0, n)[-1],
            0.606530726398907,
        )
