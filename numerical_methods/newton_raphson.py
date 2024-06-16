from numerical_methods.find_root import trunc
from numerical_methods.trail import Trail


class RaphsonTrail(Trail):
    def __init__(self, f, fd, cap, precision):
        Trail.__init__(self, f, ["x", "f(x)", "f'(x)"], cap, precision)
        self.fd = fd
        self.fdcap = lambda x: cap(fd(x), precision)

    def record(self, x):
        self.steps.append([x, self.fcap(x), self.fdcap(x)])

    def points(self, i):
        x1, x2 = self[i - 1][0], self[i][0]
        y1, y2 = self.f(x1), 0
        return [(x1, y1), (x2, y2)]

    @property
    def bounds(self):
        return (self[-1][0], self[0][0])


def newton_raphson(f, fd, x, tolerance=None, max_iterations=None, truncate=True, precision=4):
    cap = trunc if truncate else round
    trail = RaphsonTrail(f, fd, cap, precision)

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
