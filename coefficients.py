from functools import reduce
from operator import add, mul
from itertools import zip_longest, repeat

from numbers import Number
from fractions import Fraction
from collections import defaultdict

import re

class Coefficient():
    def __init__(self, *base_coefs):
        #self.cts = defaultdict(lambda: 0, cts_dict)
        assert all(type(b) is BaseCoef for b in base_coefs), "all elements must be of type BaseCoef"
        self.base_coefs = defaultdict(lambda:  BaseCoef(0), ((base.id, base) for base in base_coefs))
        
        if '' in self.base_coefs and self.base_coefs[''].coef == 0:
            del self.base_coefs['']

    def __setitem__(self, k, v):
        self.base_coefs[k] = v

    def __repr__(self):
        return ' '.join(sorted(f"{'-' if b.coef < 0 else '+'} {abs(b.coef)} {b.id}" for b in self.base_coefs.values()))
          
    def __getitem__(self, k):
        return self.base_coefs[k]

    def __bool__(self):
        return any(b for b in self.base_coefs.values())

    def __eq__(self, other):
        return type(other) is Coeficient and str(self) == str(other)

    def __add__(self, other):
        copy = self.copy()
        if not other:
            pass # skip to the end
        elif isinstance(other, Number):
            copy[''] += other
        elif type(other) is BaseCoef:
            copy[other.id] += other
        elif type(other) is Coeficient:
            for base in other.base_coefs.values():
                copy[base.id] += base
        else:
            return NotImplemented
        return copy

    def __sub__(self, other):
        return self + -1*other

    def __rsub__(self, other):
        return other + -1*self

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        copy = self.copy()
        if isinstance(other, Number) or type(other) is BaseCoef:
            for base in copy.base_coefs:
                copy[base] *= other
        elif type(other) is Coeficient:
            copy2 = copy.copy()
            copy = sum(copy2 * base for base in other.base_coefs.values())
        return copy

    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert type(other) is int
        assert other >= 0
        return reduce(mul, repeat(self, other), 1)
                        
    def copy(self):
        return Coeficient(*self.base_coefs.values())

class BaseCoef():
    def __init__(self, identifier: str, coef: Number=1):
        self.cts = defaultdict(lambda: 0)

        if coef == 0:
            self.id = ''
            self.coef = 0
            return

        if type(identifier) is str:
            assert BaseCoef.is_valid_id(identifier), f"invalid id: '{identifier}'"

            spl = identifier.split('*')
            for s in map(str.split, spl, '^'*len(spl)):
                if len(s)==1:
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
        if not other :
            return self.copy()
        if type(other) is BaseCoef:
            if not self:
                return other
            if self.id == other.id:
                return BaseCoef(self.id, self.coef+other.coef)
            return Coeficient(self, other)
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
        new_cts = dict((k, p*other) for k,p in self.cts.items())
        return BaseCoef()

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
        return f'{self.coef} {self.id}'