from functools import reduce, lru_cache
import operator
import itertools as itt
from fractions import Fraction
from numbers import Number

class vector():
    def __init__(self, iterable):
        self.v = list(iterable)

    def __repr__(self):
        return str(self.v)

    def __len__(self):
        return len(self.v)

    def __getitem__(self, i):
        return self.v[i]
        
    def __setitem__(self, i, x):
        self.v[i] = x

    def __add__(self, other):
        if type(other) is not vector:
            return NotImplemented
        assert len(other) == len(self), 'vectors must be of the same length'
        return vector([a+b for a, b in zip(self, other)])
    
    def __sub__(self, other):
        return self + -1*other

    def __mul__(self, number):
        if not isinstance(number, Number):
            return NotImplemented
        return self.scale(number)

    def __round__(self, r=0):
        return vector(round(float(i), r) for i in self.v)

    def __or__(self, other):
        assert type(other) is vector
        return self.v + other.v
    
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

    @staticmethod
    def to_fraction_unit(lst):
        n = sum(lst)
        return vector([Fraction(i,n) for i in lst]) 


class Matrix():
    def __init__(self, list_of_lists):
        assert all(len(a) == len(
            list_of_lists[0]) for a in list_of_lists[1:]), 'every row must have the same amount of elements'
        self.m = [vector(l) for l in list_of_lists]

    def __hash__(self):
        return hash(tuple(tuple(a for a in row) for row in self.m))

    def __repr__(self):
        if type(self.m[0][0]) is float:
            return '\n'.join(''.join(str(round(a, 3)).ljust(6) for a in r) for r in self.m).strip()
        return '\n'.join(''.join(str(a).ljust(8) for a in r) for r in self.m).strip()

    def __add__(self, other):
        if type(other) is not Matrix:
            return NotImplemented
        assert self.shape == other.shape, 'Matrix addition is only allowed if both matrices are the same shape'
        return Matrix([list(map(operator.add, r1, r2)) for r1, r2 in zip(self.m, other.m)])

    def __sub__(self, other):
        return self + (-1*other)

    def __mul__(self, other):
        if type(other) in (list, tuple, vector):
            assert len(
                other) == self.shape[1], 'vector size must be equal to the number of columns'
            return vector([vector.dot(r, other) for r in self.rows])
        elif type(other) is Matrix:
            assert self.shape[1] == other.shape[0], "inner shape does not match"
            return Matrix([[vector.dot(r, c) for c in other.cols] for r in self.rows])
        return NotImplemented

    def __rmul__(self, other):
        if not isinstance(other, Number):
            return NotImplemented
        return self.scale(other)

    def __eq__(self, other):
        return type(other) == Matrix and all(r1 == r2 for r1, r2 in zip(self.m, other.m))

    def __getitem__(self, i):
        return self.rows[i]
    
    def __round__(self, r=0):
        return Matrix([[round(float(i), r) for i in l] for l in self.m], precision=r)

    def __or__(self, other):
        assert type(other) in (Matrix, vector)
        assert self.shape[0] == other.shape[0]
        return Matrix([v1 | v2 for v1, v2 in zip(self.m, other.m)])

    @lru_cache(maxsize=16)
    def __pow__(self, p):
        if type(p) is not int:
            return NotImplemented
        assert self.is_square, 'Matrix exponantiation only allowed for square matrices'
        assert p >= -1, 'negative integers not allowed'
        if p == -1:
            return self.inverse
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

    @property
    def trace(self):
        return sum(r[i] for i, r in enumerate(self.m))

    @property
    def characteristic_polynomial(self):
        from polynomial import Polynomial
        m = Matrix(self.m)
        for i, row in enumerate(m):
            row[i] = Polynomial(row[i], -1)
        return m.determinant

    # @cached_property #requires python 3.8
    @property
    def determinant(self):
        assert self.is_square, "determinant is only defined for square matrices"
        s = self.shape[0]
        if s == 2:
            return self.m[0][0]*self.m[1][1] - self.m[0][1]*self.m[1][0]

        return sum((-1)**j*self.m[0][j]*self.get_minor_for(0, j).determinant for j in range(s) if self.m[0][j] != 0)

    @property
    def positive_definite(self):
        m, n = self.shape
        if (m, n) == (1,1):
            return self[0][0] > 0 
        return self.determinant > 0 and self.get_minor_for(m-1, n-1).positive_definite


    def get_minor_for(self, i, j):
        return Matrix([[x for ind_c, x in enumerate(row) if ind_c != j] for ind_r, row in enumerate(self.m) if ind_r != i])

    @staticmethod
    def identity(n):
        return Matrix([[int(i == j) for j in range(n)] for i in range(n)])

    @staticmethod
    def from_input(typ=int):
        a = input('Enter entries separated by a space:\n').strip().split(' ')
        m = [a]
        for _ in range(len(a)-1):
            m.append(input().strip().split(' '))
        return Matrix([[typ(x) for x in line] for line in m])

    # @cached_property #requires python 3.8
    @property
    def inverse(self):
        """Find the inverse of the matrix."""
        assert self.is_square, 'cannot invert non-square matrix'
        assert self.determinant != 0, 'cannot invert singular matrix'

        ans = Matrix.identity(self.shape[0])
        temp = Matrix(self.m) # copy self

        for i in range(self.shape[0]):
            op = Matrix.row_echelon_matrix(temp, i)
            ans = op * ans
            temp = op * temp
        return ans

    @staticmethod
    def row_echelon_matrix(matrix, col):
        """Preform row echelon algorithm on column col. Notice that only rows below that index will be checked."""
        n = matrix.shape[0]
        for i, r in enumerate(matrix[col:], col):
            if r[col] != 0:
                swapper = Matrix._swap_rows_matrix(n, i, col) # we use col as a row because we are on the diagonal
                return swapper * Matrix.pivot_matrix(matrix, i, col)

        print('could not eliminate row')
        return Matrix.identity(n)

    
    @staticmethod
    def pivot_matrix(matrix, row, col):
        p = matrix.m[row][col]
        assert p != 0, "can't pivot on 0 entry"
        n = matrix.shape[0]

        ans = Matrix._scale_row_matrix(n, Fraction(1, p), row)
        for i, r in enumerate(matrix.m):
            if i != row:
                ans = Matrix._add_scaled_row_matrix(n, -r[col], row, i) * ans
        return ans


    @staticmethod
    def _scale_row_matrix(size, scalar, row):
        m = Matrix.identity(size)
        m.m[row] = m.m[row].scale(scalar)
        return m

    @staticmethod
    def _swap_rows_matrix(size, i, j):
        m = Matrix.identity(size)
        m.m[i], m.m[j] = m.m[j], m.m[i]
        return m
    
    @staticmethod
    def _add_scaled_row_matrix(size, s, i, j):
        """Subtract s times row i to row j."""
        m = Matrix.identity(size)
        m.m[j][i] = s
        return m


    @staticmethod
    def least_squares(A, b):
        """Return best x such that Ax = b."""
        return (A.T*A)**-1 * A.T*b