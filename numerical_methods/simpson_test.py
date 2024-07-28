import unittest

import numpy as np
from scipy import integrate

from numerical_methods.simpson import simpson_1_3, simpson_3_8


class TestSimpson(unittest.TestCase):
    def test_simpson_1_3(self):
        f, a, b = np.sin, 0, np.pi

        assert np.isclose(
            integrate.quad(f, a, b)[0],
            simpson_1_3(f, a, b, 16),
        )

    def test_simpson_1_3_slides_exercise(self):
        f, a, b = np.exp, 0, 1

        assert np.isclose(
            1.71828,
            simpson_1_3(f, a, b, 10),
        )

    def test_simpson_3_8(self):
        f, a, b = np.sin, 0, np.pi

        assert np.isclose(
            integrate.quad(f, a, b)[0],
            simpson_3_8(f, a, b, 21),
        )
