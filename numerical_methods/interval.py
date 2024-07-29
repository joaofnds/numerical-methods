class Interval:
    def __init__(self, a, b):
        assert a <= b

        self.a = a
        self.b = b

    def __repr__(self):
        return f"[{self.a}, {self.b}]"

    def __eq__(self, other):
        assert isinstance(other, Interval)
        return self.a == other.a and self.b == other.b

    def __add__(self, other):
        assert isinstance(other, Interval)
        return Interval(self.a + other.a, self.b + other.b)

    def __sub__(self, other):
        assert isinstance(other, Interval)
        return Interval(self.a - other.b, self.b - other.a)

    def __mul__(self, other):
        assert isinstance(other, Interval)

        combinations = [
            self.a * other.a,
            self.a * other.b,
            self.b * other.a,
            self.b * other.b,
        ]

        return Interval(
            min(combinations),
            max(combinations),
        )

    def __truediv__(self, other):
        assert isinstance(other, Interval)
        assert 0 not in other

        combinations = [
            self.a / other.a,
            self.a / other.b,
            self.b / other.a,
            self.b / other.b,
        ]

        return Interval(
            min(combinations),
            max(combinations),
        )

    def __contains__(self, x):
        return self.a <= x <= self.b

    def __neg__(self):
        return Interval(-self.b, -self.a)

    def __invert__(self):
        assert 0 not in self
        return Interval(1 / self.b, 1 / self.a)

    def __and__(self, other):
        assert isinstance(other, Interval)

        if self.a > other.b or self.b < other.a:
            return 0

        return Interval(
            max(self.a, other.a),
            min(self.b, other.b),
        )

    def __or__(self, other):
        assert isinstance(other, Interval)

        return Interval(
            min(self.a, other.a),
            max(self.b, other.b),
        )

    def __xor__(self, other):
        assert isinstance(other, Interval)

        return Interval(
            min(self.a, other.a),
            max(self.b, other.b),
        )

    def __len__(self):
        return self.b - self.a

    def __abs__(self):
        return max(abs(self.a), abs(self.b))

    def radius(self):
        return len(self) / 2

    def mid(self):
        return (self.a + self.b) / 2
