import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from tabulate import tabulate


class NewthonSystemTrail:
    def __init__(self, f, j):
        self.f = f
        self.j = j
        self.steps = []

    def record(self, args):
        self.steps.append(args)

    def __getitem__(self, index):
        return self.steps[index]

    def __iter__(self):
        for step in self.steps:
            yield step

    def __len__(self):
        return len(self.steps)

    def __str__(self):
        return tabulate(
            self.steps,
            floatfmt=".4f",
            showindex=True,
        )

    @property
    def result(self):
        if len(self) > 0:
            return self[-1]

    @property
    def bounds(self):
        points = [point[0] for point in self]
        return (min(points), max(points))

    def plot(self, bounds=None):
        plt.subplots()

        if bounds is None:
            bounds = self.bounds

        XS = np.meshgrid(*[np.linspace(*bounds, 40) for _ in range(len(self.steps[0]))])
        Z = [self.f(np.array(XS))[i] for i in range(len(self))]

        colors = cm.viridis(np.linspace(0, 10, len(self)))
        for i, Z0, color in zip(range(len(self)), Z, colors):
            plt.contour(*XS, Z0, levels=[0])
            x, y = self[i]
            plt.scatter(x, y, color=color, label=f"Step {i}")
            plt.text(x, y, f"Step {i}")

        plt.grid(True)
        plt.legend()
        return plt
