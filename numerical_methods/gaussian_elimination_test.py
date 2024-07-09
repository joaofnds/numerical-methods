import unittest

import numpy as np

from numerical_methods.gaussian_elimination import gaussian_elimination


class TestGaussianElimination(unittest.TestCase):
    def test_zero_division_error(self):
        A1 = np.array(
            [
                [0, 3, 2],
                [1, 4, 3],
                [0, 2, 5],
            ],
            dtype=np.float64,
        )
        b1 = np.array([5, 6, 7], dtype=np.float64)

        with self.assertRaises(ZeroDivisionError):
            gaussian_elimination(A1, b1)

    def test_first_exercise(self):
        result = gaussian_elimination(
            A=np.array(
                [
                    [2, 3, -1],
                    [4, 4, -3],
                    [2, -3, 1],
                ],
            ),
            b=np.array([5, 3, -1], dtype=np.float64),
        )

        assert np.allclose(result, [1, 2, 3])

    def test_second_exercise(self):
        result = gaussian_elimination(
            A=np.array(
                [
                    [1.5, 5.4, 3.3],
                    [4.2, 2.3, 4.5],
                    [2.7, 5.7, 7.8],
                ],
            ),
            b=np.array([10, 11.7, 8.9]),
        )

        assert np.allclose(result, [3.126, 1.712, -1.192], atol=1e-3)

    def test_third_exercise(self):
        result = gaussian_elimination(
            A=np.array(
                [
                    [1.5, 5.4, 3.3],
                    [4.2, 2.3, 4.5],
                    [2.7, 5.7, 7.8],
                ],
            ),
            b=np.array([10, 11.7, 8.9]),
            partial_pivoting=True,
        )

        assert np.allclose(result, [3.126, 1.712, -1.192], atol=1e-3)
