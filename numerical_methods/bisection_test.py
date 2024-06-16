import math
import unittest

from numerical_methods.bisection import bisection


class TestBisection(unittest.TestCase):
    def test_raises_when_no_root(self):
        with self.assertRaisesRegex(Exception, "f doesn't have a root between 1 and 2"):
            bisection(f=lambda x: x, a=1, b=2, tolerance=0.1)

    def test_first_exercise(self):
        trail = bisection(f=lambda x: x**3 - 9 * x + 5, a=0.5, b=1, tolerance=0.01)

        assert trail.result == 0.5781

        #            n      a       b       m       f(a)     f(b)     f(m)
        assert trail[0] == [0.5000, 1.0000, 0.7500, 0.6250, -3.0000, -1.3281]
        assert trail[1] == [0.5000, 0.7500, 0.6250, 0.6250, -1.3281, -0.3808]
        assert trail[2] == [0.5000, 0.6250, 0.5625, 0.6250, -0.3808, +0.1154]
        assert trail[3] == [0.5625, 0.6250, 0.5937, 0.1154, -0.3808, -0.1340]
        assert trail[4] == [0.5625, 0.5937, 0.5781, 0.1154, -0.1340, -0.0096]

    def test_second_exercise(self):
        trail = bisection(f=lambda x: math.e**x - x - 2, a=-2, b=0, max_iterations=5)

        assert trail.result == -1.8125

        #            n       a        b        x       f(a)     f(b)     f(x)
        assert trail[0] == [-2.0000, +0.0000, -1.0000, 0.1353, -1.0000, -0.6321]
        assert trail[1] == [-2.0000, -1.0000, -1.5000, 0.1353, -0.6321, -0.2768]
        assert trail[2] == [-2.0000, -1.5000, -1.7500, 0.1353, -0.2768, -0.0762]
        assert trail[3] == [-2.0000, -1.7500, -1.8750, 0.1353, -0.0762, +0.0283]
        assert trail[4] == [-1.8750, -1.7500, -1.8125, 0.0283, -0.0762, -0.0242]


if __name__ == "__main__":
    unittest.main()
