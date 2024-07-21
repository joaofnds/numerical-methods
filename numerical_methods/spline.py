import numpy as np

# https://en.wikipedia.org/wiki/Spline_(mathematics)


def linear_spline(x: np.ndarray, y: np.ndarray, x_new: np.ndarray):
    assert np.all(np.diff(x) > 0), "x must be in increasing order."

    y_new = np.zeros_like(x_new)

    for i in range(len(x) - 1):
        mask = (x_new >= x[i]) & (x_new < x[i + 1])
        a = y[i]
        b = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        y_new[mask] = a + b * (x_new[mask] - x[i])

    return y_new


def natural_cubic_spline(x: np.ndarray, y: np.ndarray):
    n = len(x) - 1
    h = np.diff(x)

    A = np.zeros((n + 1, n + 1))
    A[0, 0], A[n, n] = 1, 1

    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]

    b = np.zeros(n + 1)
    for i in range(1, n):
        b[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    c = np.linalg.solve(A, b)

    d = np.zeros(n)
    b = np.zeros(n)
    for i in range(n):
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3

    def interpolate(x_new: np.ndarray):
        indices = np.clip(np.searchsorted(x, x_new) - 1, 0, n - 1)
        dx = x_new - x[indices]
        return y[indices] + b[indices] * dx + c[indices] * dx**2 + d[indices] * dx**3

    return interpolate
