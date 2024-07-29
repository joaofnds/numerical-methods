import unittest

import numpy as np

from numerical_methods.euler import euler


class TestEuler(unittest.TestCase):
    def test_euler(self):
        a, b, y0, n = 0, 1, 1, 0.1

        def f(x, y):
            return -x * y

        assert np.allclose(
            euler(f, a, b, y0, n),
            [1.0, 1.0, 0.99, 0.9702, 0.941094, 0.90345024, 0.85827773, 0.80678106, 0.75030639, 0.69028188, 0.62815651],
        )
