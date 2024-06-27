import unittest

import numpy as np

from numerical_methods.cholesky_decomposition import cholesky_decomposition, solve_cholesky


class TestCholeskyDecompostion(unittest.TestCase):
    def test_first_exercise(self):
        A = np.array(
            [
                [1, 1, 0],
                [1, 2, -1],
                [0, -1, 3],
            ]
        )
        b = [2, 1, 5]

        L = cholesky_decomposition(A)
        assert np.allclose(L @ L.T, A)
        assert np.allclose(solve_cholesky(A, b), np.linalg.solve(A, b))
