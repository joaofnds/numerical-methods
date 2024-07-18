import numpy as np

# https://en.wikipedia.org/wiki/Spline_(mathematics)


def linear_spline(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    w = np.zeros(len(z))
    for i in range(len(x) - 1):
        mask = (z >= x[i]) & (z <= x[i + 1])
        w[mask] = y[i] + (y[i + 1] - y[i]) * (z[mask] - x[i]) / (x[i + 1] - x[i])
    return w
