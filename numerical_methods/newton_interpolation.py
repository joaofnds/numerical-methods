import numpy as np

# https://en.wikipedia.org/wiki/Newton_polynomial
# https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter17.05-Newtons-Polynomial-Interpolation.html


def divided_differences(x: np.ndarray, y: np.ndarray):
    n = len(y)
    table = np.zeros([n, n])
    table[:, 0] = y

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])

    return table


def newton_polynomial(coef: np.ndarray, x: np.ndarray, x_new: np.ndarray):
    n = len(x) - 1
    p = coef[n]

    for k in range(1, n + 1):
        p = coef[n - k] + (x_new - x[n - k]) * p
    return p


def newton_interpolation(x: np.ndarray, y: np.ndarray, x_new: np.ndarray):
    table = divided_differences(x, y)
    return newton_polynomial(table[0], x, x_new)
