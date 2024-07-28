import numpy as np


def trapezoidal(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n

    return (h / 2) * (y[0] + 2 * np.sum(y[1:n]) + y[n])
