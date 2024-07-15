import unittest

import numpy as np

from numerical_methods.newton_interpolation import divided_differences, newton_polynomial


class TestNewtonInterpolation(unittest.TestCase):
    def test_slides_example(self):
        x = np.array([-1, 0, 1, 2, 3])
        y = np.array([1, 1, 0, -1, -2])

        table = divided_differences(x, y)

        assert np.allclose(
            table,
            np.array(
                [
                    [1, 0, -1 / 2, 1 / 6, -1 / 24],
                    [1, -1, 0, 0, 0],
                    [0, -1, 0, 0, 0],
                    [-1, -1, 0, 0, 0],
                    [-2, 0, 0, 0, 0],
                ]
            ),
        )

    def test_slides_exercise(self):
        x = np.array([-1, 0, 2])
        y = np.array([4, 1, -1])

        table = divided_differences(x, y)

        assert np.allclose(
            table,
            np.array(
                [
                    [4, -3, 2 / 3],
                    [1, -1, 0],
                    [-1, 0, 0],
                ]
            ),
        )

    def test_interpolation_slides_example(self):
        x = np.array([0.2, 0.34, 0.4, 0.52, 0.6, 0.72])
        y = np.array([0.16, 0.22, 0.27, 0.29, 0.32, 0.37])

        table = divided_differences(x, y)
        assert np.allclose(
            table,
            [
                [
                    [0.1600, 0.4286, 2.0238, -17.8985, 90.3748, -279.3421],
                    [0.2200, 0.8333, -3.7037, 18.2514, -54.8831, 0.0000],
                    [0.2700, 0.1667, 1.0417, -2.6042, 0.0000, 0.0000],
                    [0.2900, 0.3750, 0.2083, 0.0000, 0.0000, 0.0000],
                    [0.3200, 0.4167, 0.0000, 0.0000, 0.0000, 0.0000],
                    [0.3700, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                ]
            ],
            atol=1e-4,
        )

        [res] = newton_polynomial(table[0], x, np.array([0.47]))
        assert res == 0.2872096653988486
