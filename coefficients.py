from functools import reduce
from operator import add, mul
from itertools import zip_longest, repeat

from numbers import Number
from fractions import Fraction
from collections import defaultdict

from functools import cached_property

import re

superscript = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


class Coefficient(Number):
    def __init__(self, *base_coefs):
        #self.cts = defaultdict(lambda: 0, cts_dict)
        assert all(
            type(b) is BaseCoef for b in base_coefs), "all elements must be of type BaseCoef"
        self.base_coefs = defaultdict(lambda:  BaseCoef(
            0), ((base.id, base) for base in base_coefs))

        self._trim()

    def __setitem__(self, k, v):
        self.base_coefs[k] = v

    # @cached_property
    def __repr__(self):
        lst = []
        for b in self.sorted_values():
            sign = '+' if b.coef > 0 else '-'
            coef = str(abs(b.coef)) if abs(b.coef) != 1 else ''
            bases = b.id.split('*')
            center = ''
            for base in bases:
                if base != '':
                    id_, power = base.split('^')
                    power = power.translate(superscript) if power != 1 else ''
                    center += id_ + power
                else:
                    id_, power = '', ''
                    coef = str(abs(b.coef))
            lst.append(''.join((sign, coef, center)))
        pretty = ' '.join(lst).removeprefix('+')
        return f"({pretty})"

    def __getitem__(self, k):
        return self.base_coefs[k]

    def __bool__(self):
        return any(b for b in self)

    def __eq__(self, other):
        return type(other) is Coefficient and str(self) == str(other)

    def __add__(self, other):
        copy = self.copy()
        if not other:
            pass  # skip to the end
        if type(other) is Coefficient:
            for base in other:
                copy[base.id] += base
        elif isinstance(other, Number):
            copy[''] += other
        elif type(other) is BaseCoef:
            copy[other.id] += other
        else:
            return NotImplemented
        copy._trim()
        return copy

    def __sub__(self, other):
        return self + -1*other

    def __rsub__(self, other):
        return other + -1*self

    def __radd__(self, other):
        return self + other

    def __delitem__(self, k):
        del self.base_coefs[k]

    def __iter__(self):
        return (v for v in self.base_coefs.values())

    def sorted_values(self):
        return (v for k, v in sorted(self.base_coefs.items()))

    def __mul__(self, other):
        new_coef = None
        if type(other) is Coefficient:
            new_coef = sum(
                self * base for base in other)
        elif isinstance(other, Number):
            new_coef = self.copy()
            for base in new_coef.base_coefs:
                new_coef[base] *= other
        elif type(other) is BaseCoef:
            new_coef = Coefficient()
            for base in self:
                new_base = base * other
                new_coef[new_base.id] = new_base
        else:
            return NotImplemented

        new_coef._trim()
        return new_coef

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert type(
            other) is int and other >= 0, "powers must be non-negative integers"
        return reduce(mul, repeat(self, other), Coefficient(BaseCoef(1)))

    def __truediv__(self, other):
        if not type(other) is Coefficient and isinstance(other, Number):
            return Fraction(1, other) * self
        if isinstance(other, BaseCoef):
            return self * other ** -1
        return NotImplemented

    def __neg__(self):
        return -1*self

    def copy(self):
        return Coefficient(*self)

    def _trim(self):
        for k, b in self.base_coefs.copy().items():
            if b.coef == 0:
                del self[k]


class BaseCoef():
    def __init__(self, identifier: str, coef: Number = 1):
        self.cts = defaultdict(lambda: 0)

        if coef == 0:
            self.id = ''
            self.coef = 0
            return
        if identifier == '':
            self.id = ''
            self.coef = coef
            return

        if type(identifier) is str:
            assert BaseCoef.is_valid_id(
                identifier), f"invalid id: '{identifier}'"

            spl = identifier.split('*')
            for s in map(str.split, spl, '^'*len(spl)):
                if len(s) == 1:
                    s.append(1)
                self.cts[s[0]] += int(s[1])
                if self.cts[s[0]] == 0:
                    del self.cts[s[0]]

        elif isinstance(identifier, Number):
            coef = identifier
        else:
            return NotImplemented

        self.coef = coef
        self.id = BaseCoef._make_id(self.cts)

    def __bool__(self):
        return self.coef != 0

    def __hash__(self):
        return hash(self.id)

    def __add__(self, other):
        if not other:
            return self.copy()
        if type(other) is BaseCoef:
            if not self:
                return other
            if self.id == other.id:
                return BaseCoef(self.id, self.coef+other.coef)
            return Coefficient(self, other)
        if isinstance(other, Number):
            return self + BaseCoef(other)
        return NotImplemented

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if type(other) is BaseCoef:
            if self.id == '' or other.id == '':
                return BaseCoef(self.id + other.id)
            return BaseCoef(f'{self.id}*{other.id}', coef=self.coef*other.coef)

        elif isinstance(other, Number):
            return BaseCoef(self.id, self.coef*other)

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + -1*other

    def __rsub__(self, other):
        return -1*self + other

    def __pow__(self, other):
        assert type(other) is int, "can only elevate to integer powers"
        new_cts = dict((k, p*other) for k, p in self.cts.items())
        coef = self.coef**other if other >= 0 else Fraction(
            1, self.coef ** -other)
        return BaseCoef(BaseCoef._make_id(new_cts), coef)

    def __truediv__(self, other):
        if isinstance(other, Number):
            return self * Fraction(1, other)
        return self * other**-1

    def __rtruediv__(self, other):
        return self**-1 * other

    def copy(self):
        return BaseCoef(self.id, self.coef)

    @staticmethod
    def is_valid_id(identifier):
        if identifier == '':
            return True
        one_coef = '[a-zA-Z]+\d*(\^-?\d+)?'
        regex = f'{one_coef}(\*{one_coef})*'
        return bool(re.fullmatch(regex, identifier))

    @staticmethod
    def _make_id(cts_dict):
        return '*'.join(f'{c}^{p}' for c, p in sorted(cts_dict.items()))

    def __repr__(self):
        return f'_{self.coef} {self.id}'
