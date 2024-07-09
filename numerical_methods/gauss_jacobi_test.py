import unittest

import numpy as np

from numerical_methods.gauss_jacobi import gauss_jacobi


class TestGaussJacobi(unittest.TestCase):
    def test_wikipedia_example(self):
        A = np.array([[2, 1], [5, 7]])
        b = [11, 13]
        x0 = [1, 1]

        trail = gauss_jacobi(A, b, x0, max_iter=25)
        assert np.allclose(trail.result, [7.1111, -3.2222])
