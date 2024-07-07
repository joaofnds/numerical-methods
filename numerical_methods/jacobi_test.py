import unittest

import numpy as np

from numerical_methods.jacobi import jacobi


class TestJacobi(unittest.TestCase):
    def test_wikipedia_example(self):
        A = np.array([[2, 1], [5, 7]])
        b = [11, 13]
        x0 = [1, 1]

        assert np.allclose(jacobi(A, b, x0, max_iter=25), [7.1111, -3.2222])
