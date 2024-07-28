import numpy as np


def simpson_1_3(f, a, b, n):
    assert n % 2 == 0, "n must be even"

    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n

    return (h / 3) * (y[0] + 4 * sum(y[1:n:2]) + 2 * sum(y[2 : n - 1 : 2]) + y[n])


def simpson_3_8(f, a, b, n):
    assert n % 3 == 0, "n must be multiple of 3"

    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n

    return (3 * h / 8) * (y[0] + 3 * sum(y[1:n:3]) + 3 * sum(y[2:n:3]) + 2 * sum(y[3:n:3]) + y[n])
