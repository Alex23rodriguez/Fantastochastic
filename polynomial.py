from itertools import zip_longest

from numbers import Number
from fractions import Fraction

superscript = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


class Polynomial():
    def __init__(self, *coefs, start=0, identifier='λ'):
        self.coefs = [0]*start + list(coefs)
        assert len(self.coefs) > 0, "coefs iterable can't be empty"
        assert all(isinstance(c, Number) for c in self.coefs)
        while len(self.coefs) > 1 and self.coefs[-1] == 0:
            self.coefs.pop()
        self.identifier = identifier

    def __call__(self, x):
        return sum(c*(x**i) for i, c in enumerate((self.coefs)))

    def __str__(self):
        ans = ''
        for i, c in enumerate(self.coefs):
            if c != 0:
                ans = f"{'   + ' if c>0 else '   - '}{abs(c) if abs(c)!=1 else ''}{self.identifier if i > 0 else ''}{str(i).translate(superscript) if i>1 else ''}" + ans
        return ans

    def __repr__(self):
        coefs = [str(c) for c in self.coefs]
        justif = max(len(c) for c in coefs) + 2
        return ''.join(f'{self.identifier}{i}'.translate(superscript).ljust(justif) for i in range(len(self.coefs))) + '\n' + ''.join((c.ljust(justif) for c in coefs))

    def __rmul__(self, other):
        if isinstance(other, Number):
            return Polynomial(*(other * c for c in self.coefs))
        return NotImplemented

    def __mul__(self, other):
        if type(other) is not Polynomial:
            return self.__rmul__(other)

        l = [0] * (len(self.coefs) + len(other.coefs) - 1)
        for i, c1 in enumerate(self.coefs):
            for j, c2 in enumerate(other.coefs):
                l[i + j] += c1*c2

        return Polynomial(*l)

    def __add__(self, other):
        if isinstance(other, Number):
            return self.__radd__(other)
        if type(other) is not Polynomial:
            return NotImplemented
        coefs = []
        for a, b in zip_longest(self.coefs, other.coefs):
            a = 0 if a is None else a
            b = 0 if b is None else b
            coefs.append(a+b)
        return Polynomial(*coefs)

    def __sub__(self, other):
        if isinstance(other, Number):
            return self.__radd__(-other)
        if type(other) is not Polynomial:
            return NotImplemented
        return self + -1*other

    def __radd__(self, other):
        if not isinstance(other, Number):
            return NotImplemented
        coefs = self.coefs
        coefs[0] += other
        return Polynomial(*coefs)

    def __rsub__(self, other):
        if not isinstance(other, Number):
            return NotImplemented
        return other + -1*self

    def __getitem__(self, i):
        return self.coefs[i]

    def __eq__(self, other):
        if type(other) != Polynomial:
            return NotImplemented
        return self.coefs == other.coefs

    def __truediv__(self, other):
        if isinstance(other, Number):
            return Fraction(1, other) * self

        if type(other) is not Polynomial:
            return NotImplemented

        if other.degree == 0:
            return self / other[0], 0

        diff = self.degree - other.degree
        if diff < 0:  # denominator polynomial is greater. we should stop.
            return 0, self
        fits = Polynomial(
            Fraction(self[self.degree], other[other.degree]), start=diff)
        num, den = (self - fits * other) / other
        return num+fits, den

    @property
    def degree(self):
        return len(self.coefs) - 1

    def find_rough_roots(self, span=100, resolution=3):
        roots = []
        x = -span
        positive = self(x) > 0
        delta = 1 / 10**resolution
        while x <= span:
            last = positive
            x += delta
            positive = self(x) > 0

            if last ^ positive:  # XOR
                roots.append(round(x, resolution))
                if len(roots) == self.degree:
                    break
        return roots
