from functools import reduce, lru_cache
import operator
import itertools as itt


class vector():
    def __init__(self, iterable):
        self.v = list(iterable)

    def __repr__(self):
        return '\n'.join(str(a) for a in self.v)

    def __len__(self):
        return len(self.v)

    def __getitem__(self, i):
        return self.v[i]

    def __mul__(self, number):
        if type(number) not in (float, int):
            return NotImplemented
        return self.scale(number)

    @property
    def shape(self):
        return len(self.v), 1

    def magnitude(self, norm=2):
        if norm == float('inf'):
            return max(self.v)
        return (sum(x**norm for x in self.v))**(1/norm)

    @staticmethod
    def dot(u, v):
        assert len(u) == len(
            v), 'dot product only works between to vectors of the same size'
        return sum(map(operator.mul, u, v))

    def scale(self, s):
        return vector(a*s for a in self.v)

    @staticmethod
    def to_unit(iterable, norm=2):
        v = vector(iterable)
        return v.scale(1/v.magnitude(norm))


class Matrix():
    def __init__(self, list_of_lists):
        assert all(len(a) == len(
            list_of_lists[0]) for a in list_of_lists[1:]), 'every row must have the same amount of elements'
        self.m = list_of_lists
        self.round = 2

    def __hash__(self):
        return hash(tuple(tuple(a for a in row) for row in self.m))

    def __repr__(self):
        return '\n'.join(''.join(str(round(a, self.round)).ljust(self.round+3) for a in r) for r in self.m).strip()

    def __add__(self, other):
        if type(other) is not Matrix:
            return NotImplemented
        assert self.shape == other.shape, 'Matrix addition is only allowed if both matrices are the same shape'
        return Matrix([list(map(operator.add, r1, r2)) for r1, r2 in zip(self.m, other.m)])

    def __mul__(self, other):
        if type(other) in (list, tuple, vector):
            assert len(
                other) == self.shape[1], 'vector size must be equal to the number of columns'
            return [vector.dot(r, other) for r in self.rows]
        elif type(other) is Matrix:
            assert self.shape[1] == other.shape[0], "inner shape does not match"
            return Matrix([[vector.dot(r, c) for c in other.cols] for r in self.rows])
        return NotImplemented

    def __rmul__(self, other):
        if type(other) not in (int, float):
            return NotImplemented
        return self.scale(other)

    def __eq__(self, other):
        return type(other) == Matrix and all(r1 == r2 for r1, r2 in zip(self.m, other.m))

    def __getitem__(self, i):
        return self.rows[i]

    @lru_cache(maxsize=16)
    def __pow__(self, p):
        if type(p) is not int:
            return NotImplemented
        assert self.is_square, 'Matrix exponantiation only allowed for square matrices'
        assert p >= 0, 'negative integers not allowed'
        if p == 0:
            return Matrix.identity(self.shape[0])
        return reduce(operator.mul, (itt.repeat(self, p)))

    def scale(self, scalar):
        return Matrix([list(map(lambda x: x*scalar, r)) for r in self.rows])

    # @cached_property #requires python 3.8
    @property
    def shape(self):
        return len(self.m), len(self.m[0])

    @property
    def is_square(self):
        a, b = self.shape
        return a == b

    @property
    def is_symmetric(self):
        return all(a == b for a, b in zip(self.m, self.cols))

    @property
    def cols(self):
        return [list(a) for a in zip(*self.m)]

    @property
    def rows(self):
        return self.m

    @property
    def T(self):
        return Matrix(self.cols)

    # @cached_property #requires python 3.8
    @property
    def determinant(self):
        assert self.is_square, "determinant is only defined for square matrices"
        s = self.shape[0]
        if s == 2:
            return self.m[0][0]*self.m[1][1] - self.m[0][1]*self.m[1][0]

        return sum((-1)**j*self.m[0][j]*self.get_minor_for(0, j).determinant for j in range(s) if self.m[0][j] != 0)

    def get_minor_for(self, i, j):
        return Matrix([[x for ind_c, x in enumerate(row) if ind_c != j] for ind_r, row in enumerate(self.m) if ind_r != i])

    @staticmethod
    def identity(n):
        return Matrix([[int(i == j) for j in range(n)] for i in range(n)])

    @staticmethod
    def form_input(typ=int):
        a = input('Enter entries separated by a space:\n').strip().split(' ')
        m = [a]
        for _ in range(len(a)-1):
            m.append(input().split(' '))
        return Matrix([[typ(x) for x in line] for line in m])
