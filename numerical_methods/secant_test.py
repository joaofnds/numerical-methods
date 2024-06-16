import unittest

from numerical_methods.secant import secant


class TestSecant(unittest.TestCase):
    def test_first_exercise(self):
        trail = secant(
            f=lambda x: x**2 + x - 6,
            x0=1.5,
            x1=1.75,
            tolerance=0.01,
            max_iterations=5,
            precision=5,
        )

        assert trail[0] == [1.50000, -2.25000]
        assert trail[1] == [1.75000, -1.18750]
        assert trail[2] == [2.02941, 0.14791]
        assert trail[3] == [1.99846, -0.00769]
        assert trail.result == 1.99846

    def test_second_exercise(self):
        trail = secant(
            f=lambda x: x**3 - 9 * x + 3,
            x0=0,
            x1=1,
            tolerance=5 * 10e-4,
            max_iterations=5,
            truncate=False,
            precision=9,
        )

        assert trail[0] == [0.000000000, 3.000000000]
        assert trail[1] == [1.000000000, -5.000000000]
        assert trail[2] == [0.375000000, -0.322265625]
        assert trail[3] == [0.331941545, 0.049101137]
        assert trail[4] == [0.337634621, -0.000222209]
        assert trail.result == 0.337634621

    def test_third_exercise(self):
        trail = secant(
            f=lambda x: x**3 - 5 * x**2 + 17 * x + 21,
            x0=-1,
            x1=1,
            max_iterations=7,
            truncate=False,
            precision=10,
        )

        assert trail[0] == [-1.0000000000, -2.0000000000]
        assert trail[1] == [1.0000000000, 34.0000000000]
        assert trail[2] == [-0.8888888889, 1.2359396430]
        assert trail[3] == [-0.9601423487, -0.8169162008]
        assert trail[4] == [-0.9317876516, 0.0094644431]
        assert trail[5] == [-0.9321123947, 0.0000712188]
        assert trail[6] == [-0.9321148569, -0.0000000068]

        assert trail.result == -0.9321148569


if __name__ == "__main__":
    unittest.main()
