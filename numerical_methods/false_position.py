from numerical_methods.find_root import find_root


def false_posittion(f, a, b, tolerance=None, max_iterations=None, truncate=True, precision=4):
    return find_root(
        f=f,
        a=a,
        b=b,
        update=lambda a, b: (a * f(b) - b * f(a)) / (f(b) - f(a)),
        tolerance=tolerance,
        max_iterations=max_iterations,
        truncate=truncate,
        precision=precision,
    )
