from random import random
from functools import reduce, lru_cache
from matrix import Matrix, vector
import itertools as itt


class MarkovChain():
    def __init__(self, trans_matrix: Matrix, init_prob=None):
        if init_prob is None:
            size = trans_matrix.shape[0]
            init_prob = [1/size for _ in range(size)]
        else:
            assert type(init_prob) in (
                list, tuple, vector), "initial probabilities must be vector-like"
            assert sum(
                init_prob) == 1, "sum of initial probabilities must equal 1"
        assert type(
            trans_matrix) is Matrix, "transition matrix must be of type matrix"
        assert trans_matrix.is_square, "transition matrix must be square"
        assert all(round(sum(
            r), 9) == 1 for r in trans_matrix.rows), "the rows of the transition matrix must add up to 1"
        self.P = trans_matrix
        self.p0 = init_prob

        self.restart()

    def restart(self):
        self.state = self._pick(self.p0)

    def next(self, n=1):
        for _ in range(n):
            self.state = self._pick(self.P[self.state])
        return self.state

    def _pick(self, prob):
        r = random()
        for i, p in enumerate(itt.accumulate(prob)):
            if r < p:
                return i

    def probXY(self, x, y, n=1):
        """Probability of transitioning from state x to state y in n timesteps"""
        m = self.P ** n
        return m[x][y]


    def probX(self, x, n=1):
        """Probability of being at state x in the n-th timestep."""
        return self.prob(n)[x]

    def prob(self, n=1):
        """Probability of all states at timestep n."""
        m = self.P ** n
        return vector(m.T*self.p0)

    def prob_Xn_eq_x_given_Xn2_eq_y(self, n, x, n2, y):
        diff = n - n2
        if diff >= 0:
            return self.probXY(y, x, diff)
        # apply bayes: P(A|B) = P(B|A)P(A)/P(B)
        return self.probXY(y, x, -diff) * self.probX(x, n) / self.probX(y, n2)
        

    @staticmethod
    def from_unscaled_matrix(unscaled_matrix: Matrix, init_prob=None):
        if init_prob is not None:
            assert any(p != 0 for p in init_prob)
            init_prob = vector.to_unit(init_prob, norm=1)
        assert all(any(p != 0 for p in row) for row in unscaled_matrix)
        scaled_matrix = Matrix([vector.to_unit(r, norm=1)
                                for r in unscaled_matrix])
        return MarkovChain(scaled_matrix, init_prob)

    @staticmethod
    def from_unscaled_to_fraction(unscaled_matrix: Matrix, init_prob=None):
        if init_prob is not None:
            assert any(p != 0 for p in init_prob)
            init_prob = vector.to_fraction_unit(init_prob)
        assert all(any(p != 0 for p in row) for row in unscaled_matrix)
        scaled_matrix = Matrix([vector.to_fraction_unit(r)
                                for r in unscaled_matrix])
        return MarkovChain(scaled_matrix, init_prob)


    # @cached_property
    @property
    @lru_cache(maxsize=1)
    def classes(self):
        """Calculate the communication classes of the MC."""
        classes = {}
        for i in range(len(self.p0)):
            for j in range(i, len(self.p0)):
                if self.accessible(i, j) and self.accessible(j, i):
                    # communicating!
                    if i in classes:
                        classes[i].add(j)
                    else:
                        classes[i] = set((i, j))
                    classes[j] = classes[i]
        m = map(tuple, classes.values())
        return dict((t, self._recurrent(t[0])) for t in m)

    def accessible(self, i, j, tried=None):
        """Indicate if i->j."""
        if tried is None:
            tried = set()
        tried.add(i)
        if i == j:
            return True
        access = [k for k, a in enumerate(
            self.P[i]) if a > 0 and k not in tried]
        for k in access:
            if self.accessible(k, j, tried):
                return True
        return False

    def _recurrent(self, i):
        is_recurrent = all(self.accessible(j, i)
                           for j, a in enumerate(self.P[i]) if a > 0)
        return 'recurrent' if is_recurrent else 'transitive'
