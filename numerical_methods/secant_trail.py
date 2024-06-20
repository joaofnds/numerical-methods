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
