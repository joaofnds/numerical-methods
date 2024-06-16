import unittest

from numerical_methods.false_position import false_posittion


class TestFalsePosition(unittest.TestCase):
    def test_raises_when_no_root(self):
        with self.assertRaisesRegex(Exception, "doesn't have a root between 1 and 2"):
            false_posittion(f=lambda x: x, a=1, b=2, tolerance=0.1)

    def test_first_exercise(self):
        trail = false_posittion(
            f=lambda x: x**3 - 9 * x + 5,
            a=0,
            b=1,
            tolerance=0.0005,
        )

        assert trail.result == 0.5769

        #            n      a       b       x       f(a)     f(b)     f(x)
        assert trail[0] == [0.0000, 1.0000, 0.6250, 5.0000, -3.0000, -0.3808]
        assert trail[1] == [0.0000, 0.6250, 0.5807, 5.0000, -0.3808, -0.0304]
        assert trail[2] == [0.0000, 0.5807, 0.5771, 5.0000, -0.0304, -0.0017]
        assert trail[3] == [0.0000, 0.5771, 0.5769, 5.0000, -0.0017, +0.0000]

    def test_second_exercise(self):
        trail = false_posittion(
            f=lambda x: 2 * x**3 + 5 * x**2 - 8 * x - 10,
            a=0,
            b=3,
            tolerance=0.05,
            precision=6,
        )

        assert trail.result == 1.672066

        assert trail[0] == [0.000000, 3.000000, 0.400000, -10.000000, 65.000000, -12.272000]
        assert trail[1] == [0.400000, 3.000000, 0.812920, -12.272000, 65.000000, -12.124747]
        assert trail[2] == [0.812920, 3.000000, 1.156749, -12.124747, 65.000000, -9.468032]
        assert trail[3] == [1.156749, 3.000000, 1.391104, -9.468032, 65.000000, -6.068933]
        assert trail[4] == [1.391104, 3.000000, 1.528495, -6.068933, 65.000000, -3.404438]
        assert trail[5] == [1.528495, 3.000000, 1.601730, -3.404438, 65.000000, -1.767543]
        assert trail[6] == [1.601730, 3.000000, 1.638746, -1.767543, 65.000000, -0.880858]
        assert trail[7] == [1.638746, 3.000000, 1.656946, -0.880858, 65.000000, -0.430026]
        assert trail[8] == [1.656946, 3.000000, 1.665772, -0.430026, 65.000000, -0.207838]
        assert trail[9] == [1.665772, 3.000000, 1.670024, -0.207838, 65.000000, -0.099963]
        assert trail[10] == [1.670024, 3.000000, 1.672066, -0.099963, 65.000000, -0.047964]


if __name__ == "__main__":
    unittest.main()
