import numpy as np

# https://en.wikipedia.org/wiki/Lagrange_polynomial
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.lagrange.html
# https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter17.04-Lagrange-Polynomial-Interpolation.html


def lagrange(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    n, m = len(x), len(z)
    w = np.zeros(m)

    for i in range(n):
        p = np.ones(m)
        for j in range(n):
            if i != j:
                p *= (z - x[j]) / (x[i] - x[j])
        w += y[i] * p

    return w
