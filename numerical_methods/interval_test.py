import unittest
from math import sqrt

from numerical_methods.interval import Interval


class TestInterval(unittest.TestCase):
    def test_addition(self):
        a, b = -3, 4.9
        c, d = 1.5, 3.7
        self.assertEqual(
            Interval(a, b) + Interval(c, d),
            Interval(a + c, b + d),
        )

    def test_subtraction(self):
        a, b = -3, 4.9
        c, d = 1.5, 3.7
        self.assertEqual(
            Interval(a, b) - Interval(c, d),
            Interval(a - d, b - c),
        )

    def test_multiplication(self):
        a, b = -2, 3
        c, d = -3, 6
        self.assertEqual(
            Interval(a, b) * Interval(c, d),
            Interval(a * d, b * d),
        )

    def test_division(self):
        a, b = -4, 5
        c, d = 1.5, 2.6

        self.assertEqual(
            Interval(a, b) / Interval(c, d),
            Interval(a / c, b / c),
        )

    def test_contains(self):
        interval = Interval(-10, 10)
        for i in range(-10, 11):
            self.assertIn(i, interval)

    def test_negation(self):
        self.assertEqual(
            -Interval(1, 2),
            Interval(-2, -1),
        )

    def test_inversion(self):
        a, b = sqrt(2), 5

        self.assertEqual(
            ~Interval(a, b),
            Interval(1 / b, 1 / a),
        )

    def test_intersection(self):
        self.assertEqual(
            Interval(-2, 4) & Interval(2, 6),
            Interval(2, 4),
        )

    def test_union(self):
        self.assertEqual(
            Interval(-2, 4) | Interval(2, 6),
            Interval(-2, 6),
        )

    def test_convex_set(self):
        self.assertEqual(
            Interval(-2, 1) ^ Interval(2, 6),
            Interval(-2, 6),
        )

    def test_length(self):
        self.assertEqual(
            len(Interval(-2, 4)),
            6,
        )

    def test_radius(self):
        self.assertEqual(
            Interval(-2, 4).radius(),
            3,
        )

    def test_midpoint(self):
        self.assertEqual(
            Interval(-2, 4).mid(),
            1,
        )

    def test_modulus(self):
        self.assertEqual(
            abs(Interval(-2, 4)),
            4,
        )
