from numerical_methods.find_root import trunc
from numerical_methods.newton_raphson_trail import NewtonRaphsonTrail


def newton_raphson(f, fd, x, tolerance=None, max_iterations=None, truncate=True, precision=4):
    cap = trunc if truncate else round
    trail = NewtonRaphsonTrail(f, fd, cap, precision)

    i = 1
    while True:
        trail.record(x)

        if tolerance is not None and abs(f(x)) < tolerance:
            break

        if max_iterations is not None and i >= max_iterations:
            break

        x = cap(x - (f(x) / fd(x)), precision)

        i += 1

    return trail
