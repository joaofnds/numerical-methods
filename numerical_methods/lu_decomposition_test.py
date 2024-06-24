import unittest

import numpy as np

from numerical_methods.lu_decomposition import lu_decomposition, solve_lu


class TestLUDecomposition(unittest.TestCase):
    def test_first_exercise(self):
        A = np.array(
            [
                [3, 2, 4],
                [1, 1, 2],
                [4, 3, -2],
            ]
        )
        b = [1, 2, 3]

        L, U = lu_decomposition(A)

        assert np.allclose(
            L,
            [
                [1, 0, 0],
                [1 / 3, 1, 0],
                [4 / 3, 1, 1],
            ],
        )

        assert np.allclose(
            U,
            [
                [3, 2, 4],
                [0, 1 / 3, 2 / 3],
                [0, 0, -8],
            ],
        )

        assert np.allclose(
            solve_lu(L, U, b),
            np.linalg.solve(A, b),
        )
