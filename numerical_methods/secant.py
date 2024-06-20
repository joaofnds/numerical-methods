from numerical_methods.find_root import trunc
from numerical_methods.secant_trail import SecantTrail


def secant(f, x0, x1, tolerance=None, max_iterations=None, truncate=True, precision=4):
    cap = trunc if truncate else round
    trail = SecantTrail(f, cap, precision)
    trail.record(x0)

    i = 2
    while True:
        trail.record(x1)

        if tolerance is not None and abs(f(x1)) < tolerance:
            break

        if max_iterations is not None and i >= max_iterations:
            break

        result = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0, x1 = x1, cap(result, precision)

        i += 1

    return trail
