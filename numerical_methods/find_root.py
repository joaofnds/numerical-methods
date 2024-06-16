from numerical_methods.trail import Trail


class TwoPointTrail(Trail):
    def __init__(self, f, cap, precision):
        Trail.__init__(self, f, ["a", "b", "x", "f(a)", "f(b)", "f(x)"], cap, precision)

    def record(self, a, b, x):
        self.steps.append([a, b, x, self.fcap(a), self.fcap(b), self.fcap(x)])

    @property
    def result(self):
        return self[-1][2]

    def points(self, i):
        x1, x2 = self[i][0], self[i][1]
        y1, y2 = self.f(x1), self.f(x2)
        return [(x1, y1), (x2, y2)]

    @property
    def bounds(self):
        return (self[0][0], self[0][1])


def trunc(x, digits):
    n = 10**digits
    return int(x * n) / float(n)


def find_root(f, a, b, update, tolerance=None, max_iterations=None, truncate=True, precision=4):
    cap = trunc if truncate else round
    trail = TwoPointTrail(f, cap, precision)

    i = 1
    while True:
        if f(a) * f(b) >= 0:
            raise Exception(f"f doesn't have a root between {a} and {b}")

        x = cap(update(a, b), precision)

        trail.record(a, b, x)

        if tolerance is not None and abs(f(x)) < tolerance:
            break

        if max_iterations is not None and i >= max_iterations:
            break

        if f(a) * f(x) >= 0:
            a = x
        else:
            b = x

        i += 1

    return trail
