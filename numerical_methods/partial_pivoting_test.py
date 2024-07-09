from unittest import TestCase

import numpy as np

from numerical_methods.partial_pivoting import partial_pivoting


class TestPartialPivoting(TestCase):
    def test_two_by_two(self):
        A = np.array(
            [
                [1, 2],
                [3, 4],
            ]
        )
        b = np.array([1, 2])

        P, Ap, bp = partial_pivoting(A, b)

        assert np.allclose(
            P,
            [
                [0, 1],
                [1, 0],
            ],
        )

        assert np.allclose(
            Ap,
            [
                [3, 4],
                [1, 2],
            ],
        )
        assert np.allclose(bp, [2, 1])

    def test_three_by_three(self):
        A = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )
        b = np.array([1, 2, 3])

        P, Ap, bp = partial_pivoting(A, b)

        assert np.allclose(
            P,
            [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ],
        )

        assert np.allclose(
            Ap,
            [
                [7, 8, 9],
                [4, 5, 6],
                [1, 2, 3],
            ],
        )
        assert np.allclose(bp, [3, 2, 1])

    def test_four_by_four(self):
        A = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
            ]
        )
        b = np.array([1, 2, 3, 4])

        P, Ap, bp = partial_pivoting(A, b)

        assert np.allclose(
            P,
            [
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ],
        )

        assert np.allclose(
            Ap,
            [
                [13, 14, 15, 16],
                [9, 10, 11, 12],
                [5, 6, 7, 8],
                [1, 2, 3, 4],
            ],
        )
        assert np.allclose(bp, [4, 3, 2, 1])
