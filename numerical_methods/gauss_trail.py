import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from tabulate import tabulate


class GaussTrail:
    def __init__(self):
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
        return (min(map(min, self.steps)), max(map(max, self.steps)))

    def plot(self, F, headers, bounds=None):
        plt.subplots()

        (a, b) = self.bounds if bounds is None else bounds
        pad = (b - a) * 0.1
        x = np.arange(a - pad, b + pad, 0.01)

        x = np.linspace(a, b, 100)
        colors = cm.viridis(np.linspace(0, len(self), len(self) + len(self[0])))

        for y, header, color in zip(F(x), headers, colors):
            plt.plot(x, y, label=header, color=color)

        for i, (x, y), color in zip(range(len(self)), self, colors):
            plt.scatter(x, y, color=color)
            plt.text(x, y, i)

        plt.grid(True)
        plt.legend()
        return plt
