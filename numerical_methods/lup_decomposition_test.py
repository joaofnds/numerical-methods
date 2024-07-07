import unittest

import numpy as np

from numerical_methods.lup_decomposition import lup_decomposition, solve_lup


class TestLUDecomposition(unittest.TestCase):
    def test_first_exercise(self):
        A = np.array(
            [
                [2, 1, -1],
                [-3, -1, 2],
                [-2, 1, 2],
            ],
            dtype=np.float64,
        )
        b = np.array([1, 2, 3], dtype=np.float64)

        L, U, P = lup_decomposition(A)
        assert np.allclose(P @ A, L @ U)

        assert np.allclose(solve_lup(A, b), np.linalg.solve(A, b))

    def test_second_exercise(self):
        A = np.array(
            [
                [1, 2, 0],
                [3, 4, 4],
                [5, 6, 3],
            ],
            dtype=np.float64,
        )
        b = np.array([3, 7, 8], dtype=np.float64)

        L, U, P = lup_decomposition(A)
        assert np.allclose(P @ A, L @ U)

        assert np.allclose(solve_lup(A, b), np.linalg.solve(A, b))
