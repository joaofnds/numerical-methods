from numerical_methods.trail import Trail


class NewtonRaphsonTrail(Trail):
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
