import pygrank.algorithms.utils
import scipy
from numpy.linalg import norm
from numpy import dot
from numpy.random import choice
import numpy as np
from pygrank.algorithms.utils import optimize
from pygrank.algorithms.abstract_filters import ClosedFormGraphFilter


class GenericGraphFilter(ClosedFormGraphFilter):
    """Implements a graph filter with a specific vector of weight parameters."""

    def __init__(self, weights=[0.9]*10, **kwargs):
        """
        Initializes the graph filter.
        Args:
            weights: Optional. A list-like object with elements weights[n] proportional to the importance of propagating
                personalization graph signals n hops away. Default is [0.9]*10 .

        Example:
            >>> from pygrank.algorithms import learnable
            >>> algorithm = learnable.GenericGraphFilter([0.5, 0.25, 0.125], tol=1.E-9) # tol passed to the ConvergenceManager
        """
        super(GenericGraphFilter, self).__init__(**kwargs)
        self.weights = weights

    def _coefficient(self, _):
        if self.convergence.iteration >= len(self.weights):
            return 0
        return self.weights[self.convergence.iteration]


class LanczosFilter:

    def __init__(self, weights=None, krylov_space_degree=None, to_scipy=None, fraction_of_training=.5, **kwargs):
        self.weights = weights
        self.krylov_space_degree = (10 if weights is None else len(weights)) if krylov_space_degree is None else krylov_space_degree
        self.to_scipy = pygrank.algorithms.utils.preprocessor(**kwargs) if to_scipy is None else to_scipy
        self.fraction_of_training = fraction_of_training

    def _extract_krylov_space_base(self, M, s):
        base = [s/norm(s,2)]
        base_norms = []
        alphas = []

        for j in range(0,self.krylov_space_degree):
            v = base[j]
            w = v*M
            a = dot(v, w)
            next_v = w-v*a
            if j > 0:
                next_v -= base[j-1]*base_norms[j-1]
            next_v_norm = norm(next_v, 2)
            base_norms.append(next_v_norm)
            if j != self.krylov_space_degree-1:
                base.append(next_v/next_v_norm)
            alphas.append(a)
        H = scipy.sparse.diags([alphas, base_norms, base_norms], [0, -1, 1])
        return base, H

    def _rank_from_lanczos_decomposition(self, V, H, weights):
        filterH = 0
        powH = 1*H
        for weight in weights:
            if weight != 0:
                filterH += weight*powH
            powH *= H
        if sum(weights)!=0:
            filterH /= sum(weights)
        e1 = np.repeat(0.0, self.krylov_space_degree)
        e1[0] = 1
        ranks = (V*filterH)@e1
        if ranks.sum() != 0:
            ranks = ranks / ranks.sum()
        return ranks

    def rank(self, G, personalization=None, **kwargs):
        M = self.to_scipy(G)
        #degrees = scipy.array(M.sum(axis=1)).flatten()
        #M = scipy.sparse.diags(scipy.repeat(1.0, len(G))) - M
        is_known = np.repeat(1.0, len(G)) if personalization is None else np.array([1 if n in personalization else 0 for n in G], dtype=float)
        personalization = np.repeat(1.0, len(G)) if personalization is None else np.array([personalization.get(n, 0) for n in G], dtype=float)
        if personalization.sum() == 0:
            raise Exception("The personalization vector should contain at least one non-zero entity")

        if self.weights is None:
            if self.fraction_of_training == 1:
                training_choice = 1
                test_choice = 1
            else:
                training_choice = choice([1,0], size=personalization.size, p=[self.fraction_of_training, 1-self.fraction_of_training])
                test_choice = 1-training_choice
            if test_choice.sum()==0:
                raise Exception("Empty validation set")
            base, H = self._extract_krylov_space_base(M, training_choice*personalization)
            V = np.column_stack(base)
            loss = lambda weights: norm((self._rank_from_lanczos_decomposition(V, H, weights)-personalization)*test_choice*is_known,2)
            weights = optimize(loss, divide_range=1.5)
        else:
            weights = self.weights

        base, H = self._extract_krylov_space_base(M, personalization)
        V = np.column_stack(base)
        ranks = self._rank_from_lanczos_decomposition(V, H, weights)

        ranks = dict(zip(G.nodes(), map(float, ranks)))
        return ranks