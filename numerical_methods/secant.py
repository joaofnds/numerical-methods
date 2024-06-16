from numerical_methods.find_root import trunc
from numerical_methods.trail import Trail


class SecantTrail(Trail):
    def __init__(self, f, cap, precision):
        Trail.__init__(self, f, ["x", "f(x)"], cap, precision)

    def record(self, x):
        self.steps.append([x, self.fcap(x)])

    def points(self, i):
        x1, x2 = self[i - 1][0], self[i][0]
        y1, y2 = self[i - 1][1], self[i][1]
        return [(x1, y1), (x2, y2)]

    @property
    def bounds(self):
        points = [point[0] for point in self]
        return (min(points), max(points))


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
