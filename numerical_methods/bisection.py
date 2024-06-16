from numerical_methods.find_root import find_root


def bisection(f, a, b, tolerance=None, max_iterations=None, truncate=True, precision=4):
    return find_root(
        f=f,
        a=a,
        b=b,
        update=lambda a, b: (a + b) / 2,
        tolerance=tolerance,
        max_iterations=max_iterations,
        truncate=truncate,
        precision=precision,
    )
