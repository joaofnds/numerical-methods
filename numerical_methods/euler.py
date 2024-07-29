import numpy as np


def euler(f, x0, xf, y0, h):
    t = np.arange(x0, xf + h, h)
    y = np.zeros(t.size)
    y[0] = y0
    for i in range(1, t.size):
        y[i] = y[i - 1] + h * f(t[i - 1], y[i - 1])
    return y
