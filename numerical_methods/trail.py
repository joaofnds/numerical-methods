import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from tabulate import tabulate


class Trail:
    def __init__(self, f, headers, cap, precision):
        self.cap = lambda x: cap(x, precision)
        self.f = f
        self.fcap = lambda x: self.cap(f(x))
        self.headers = headers
        self.precision = precision
        self.steps = []

    def record(self, *args):
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
            headers=self.headers,
            floatfmt=f".{self.precision}f",
            showindex=True,
        )

    @property
    def result(self):
        assert len(self) > 0
        return self[-1][0]

    def relative_error(self, true_root):
        return abs(self.result - true_root) / abs(true_root)

    def plot(self, bounds=None):
        plt.subplots()

        (a, b) = self.bounds if bounds is None else bounds
        pad = (b - a) * 0.1
        x = np.arange(a - pad, b + pad, 0.01)

        plt.plot(x, self.f(x), label="f(x)")

        colors = cm.viridis(np.linspace(0, 1, len(self)))
        for i, color in zip(range(1, len(self)), colors):
            [(x1, y1), (x2, y2)] = self.points(i)

            if x1 != x2:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1

                plt.plot(x, x * slope + intercept, label=i, color=color)
                plt.scatter([x1, x2], [y1, y2], color=color)

        plt.grid(True)
        plt.legend()
        plt.xlim(x.min(), x.max())
