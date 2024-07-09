import unittest

import numpy as np

from numerical_methods.gauss_seidel import gauss_seidel


class TestGaussSeidel(unittest.TestCase):
    def test_wikipedia_first_example(self):
        A = np.array(
            [
                [16, 3],
                [7, -11],
            ],
            dtype=np.float64,
        )
        b = np.array([11, 13], dtype=np.float64)

        trail = gauss_seidel(A, b, max_iter=5)
        assert np.allclose(trail.result, [0.8122, -0.6650], atol=1e-4)

    def test_wikipedia_second_example(self):
        A = np.array(
            [
                [10, -1, 2, 0],
                [-1, 11, -1, 3],
                [2, -1, 10, -1],
                [0, 3, -1, 8],
            ],
            dtype=np.float64,
        )
        b = np.array([6, 25, -11, 15], dtype=np.float64)

        trail = gauss_seidel(A, b, max_iter=6)
        assert np.allclose(trail.result, [1, 2, -1, 1])
