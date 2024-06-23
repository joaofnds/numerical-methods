import unittest

import numpy as np

from numerical_methods.gaussian_elimination import gaussian_elimination


class TestGaussianElimination(unittest.TestCase):
    def test_first_exercise(self):
        result = gaussian_elimination(
            A=[
                [2, 3, -1],
                [4, 4, -3],
                [2, -3, 1],
            ],
            b=[5, 3, -1],
        )

        assert result == [1, 2, 3]

    def test_second_exercise(self):
        result = gaussian_elimination(
            A=[
                [1.5, 5.4, 3.3],
                [4.2, 2.3, 4.5],
                [2.7, 5.7, 7.8],
            ],
            b=[10, 11.7, 8.9],
        )

        assert np.allclose(result, [3.126, 1.712, -1.192], atol=1e-3)
