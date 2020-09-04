import pygrank.algorithms.utils
import scipy
from numpy.linalg import norm
from numpy import dot
from numpy.random import choice
import numpy as np


class LanczosFilter:
    # https://arxiv.org/pdf/1509.04537.pdf

    def __init__(self, weights=None, krylov_space_degree=None, to_scipy=None, fraction_of_training=1, **kwargs):
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
            w = M*v
            a = dot(v, w)
            next_v = w-v*a
            if j > 0:
                next_v -= base[j-1]*base_norms[j-1]
            next_v_norm = norm(next_v,2)
            base_norms.append(next_v_norm)
            if j!=self.krylov_space_degree-1:
                base.append(next_v/next_v_norm)
            alphas.append(a)
        H = scipy.sparse.diags([alphas, base_norms, base_norms], [0,-1,1])
        return base, H

    def rank(self, G, personalization=None, **kwargs):
        M = self.to_scipy(G)
        #degrees = scipy.array(M.sum(axis=1)).flatten()
        #M = scipy.sparse.diags(scipy.repeat(1.0, len(G))) - M
        personalization = scipy.repeat(1.0, len(G)) if personalization is None else scipy.array([personalization.get(n, 0) for n in G], dtype=float)
        if self.fraction_of_training == 1:
            training_choice = 1
            test_choice = 1
        else:
            training_choice = choice([1,0], size=personalization.size, p=[self.fraction_of_training, 1-self.fraction_of_training])
            test_choice = 1-training_choice

        training_personalization = training_choice*personalization
        test_personalization = test_choice*personalization
        base, H = self._extract_krylov_space_base(M, training_personalization)

        filterH = 0
        powH = 1*H
        for weight in self.weights:
            filterH += weight*powH
            powH *= H

        e1 = scipy.repeat(0.0, self.krylov_space_degree)
        e1[0] = 1
        ranks = (np.column_stack(base)*filterH)@e1
        ranks = ranks/ranks.sum()


        """
        ranks = 0
        for base_vector in base:
            sim = dot(base_vector, test_personalization)
            ranks += sim*base_vector
        """

        ranks = dict(zip(G.nodes(), map(float, ranks)))
        return ranks