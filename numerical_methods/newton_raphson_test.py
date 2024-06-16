import unittest

import numpy as np

from numerical_methods.newton_raphson import newton_raphson


class TestNewtonRaphson(unittest.TestCase):
    def test_first_exercise(self):
        trail = newton_raphson(
            f=lambda x: x**3 - 9 * x + 3,
            fd=lambda x: 3 * x**2 - 9,
            x=0.75,
            tolerance=0.01,
            max_iterations=4,
            precision=3,
        )

        assert trail[0] == [0.750, -3.328, -7.312]
        assert trail[1] == [0.294, 0.379, -8.740]
        assert trail[2] == [0.337, 0.005, -8.659]
        assert trail.result == 0.337

    def test_second_exercise(self):
        trail = newton_raphson(
            f=lambda x: 2 * x**3 + np.log(x) - 5,
            fd=lambda x: 6 * x**2 + 1 / x,
            x=2,
            tolerance=10e-7,
            max_iterations=6,
            truncate=False,
            precision=5,
        )

        assert trail[0] == [2.00000, 11.69315, 24.50000]
        assert trail[1] == [1.52273, 2.48203, 14.56896]
        assert trail[2] == [1.35237, 0.24857, 11.71287]
        assert trail[3] == [1.33115, 0.00353, 11.38299]
        assert trail[4] == [1.33084, 0.00001, 11.37822]
        assert trail[5] == [1.33084, 0.00001, 11.37822]
        assert trail.result == 1.33084


if __name__ == "__main__":
    unittest.main()
