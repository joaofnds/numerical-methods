import numpy as np


def newton_cotes(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n

    return h * (y[0] / 2 + np.sum(y[1:-1]) + y[-1] / 2)
