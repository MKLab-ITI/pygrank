import scipy
import numpy as np
import warnings
import pygrank.algorithms.utils


class PageRank:
    """A Personalized PageRank power method algorithm. Supports warm start."""

    def __init__(self, alpha=0.85, normalization='auto', convergence=None):
        """ Initializes the PageRank scheme parameters.

        Attributes:
            alpha: Optional. 1-alpha is the bias towards the personalization. Default value is 0.85.
            normalization: Optional. The normalization parameter used by pygrank.algorithms.utils.to_scipy_sparse_matrix.
            convergence: Optional. The ConvergenceManager that determines when iterations stop. If None (default),
                the default ConvergenceManager() is used.
        """
        self.alpha = float(alpha) # typecast to make sure that a graph is not accidentally the first argument
        self.normalization = normalization
        self.convergence = pygrank.algorithms.utils.ConvergenceManager(max_iters=int(10.0/(1.-alpha))) if convergence is None else convergence

    def rank(self, G, personalization=None, warm_start=None):
        M = pygrank.algorithms.utils.to_scipy_sparse_matrix(G, self.normalization)
        degrees = scipy.array(M.sum(axis=1)).flatten()

        personalization = scipy.repeat(1.0, len(G)) if personalization is None else scipy.array([personalization.get(n, 0) for n in G], dtype=float)
        personalization = personalization / personalization.sum()
        ranks = personalization if warm_start is None else scipy.array([warm_start.get(n, 0) for n in G], dtype=float)

        is_dangling = scipy.where(degrees == 0)[0]
        self.convergence.start()
        while not self.convergence.has_converged(ranks):
            ranks = self.alpha * (ranks * M + sum(ranks[is_dangling]) * personalization) + (1 - self.alpha) * personalization
            ranks = ranks/ranks.sum()

        ranks = dict(zip(G.nodes(), map(float, ranks)))
        return ranks


class HeatKernel:
    """ Heat kernel filter."""

    def __init__(self, t=5, normalization='auto', convergence=None):
        self.t = t
        self.normalization = normalization
        self.convergence = pygrank.algorithms.utils.ConvergenceManager() if convergence is None else convergence

    def rank(self, G, personalization=None):
        M = pygrank.algorithms.utils.to_scipy_sparse_matrix(G, self.normalization)

        personalization = scipy.repeat(1.0, len(G)) if personalization is None else scipy.array([personalization.get(n, 0) for n in G], dtype=float)
        personalization = personalization / personalization.sum()

        coefficient = np.exp(-self.t)
        ranks = personalization*coefficient

        self.convergence.start()
        Mpower = M
        while not self.convergence.has_converged(ranks):
            coefficient *= self.t/(self.convergence.iteration+1)
            Mpower *= M
            ranks += personalization*Mpower

        ranks = dict(zip(G.nodes(), map(float, ranks)))
        return ranks


class BiasedKernel:
    """ Heuristic kernel-like method that places emphasis on shorter random walks."""

    def __init__(self, alpha=0.85, t=5, normalization='auto', convergence=None):
        self.alpha = alpha
        self.normalization = normalization
        self.convergence = pygrank.algorithms.utils.ConvergenceManager() if convergence is None else convergence
        warnings.warn("BiasedKernel is still under development (its implementation may be incorrect)", stacklevel=2)
        warnings.warn("BiasedKernel is a low-quality heuristic", stacklevel=2)

    def rank(self, G, personalization=None, warm_start=None):
        M = pygrank.algorithms.utils.to_scipy_sparse_matrix(G, self.normalization)
        degrees = scipy.array(M.sum(axis=1)).flatten()

        personalization = scipy.repeat(1.0, len(G)) if personalization is None else scipy.array([personalization.get(n, 0) for n in G], dtype=float)
        personalization = personalization / personalization.sum()
        ranks = personalization if warm_start is None else scipy.array([warm_start.get(n, 0) for n in G], dtype=float)

        is_dangling = scipy.where(degrees == 0)[0]
        self.convergence.start()
        while not self.convergence.has_converged(ranks):
            a = self.alpha*self.t/self.convergence.iteration
            ranks = np.exp(-self.t) * personalization + a * ((ranks * M + sum(ranks[is_dangling]) * personalization) - ranks)
            ranks = ranks/ranks.sum()

        ranks = dict(zip(G.nodes(), map(float, ranks)))
        return ranks


class Fast:
    """ Fast computation of PageRank with progressively lower restart probabilities (relies on warm start)."""

    def __init__(self, ranker, enabled=True):
        self.ranker = ranker
        self.enabled = enabled
        # warnings.warn("Fast implementation of PageRank still under development (could be slower)", stacklevel=2)

    def rank(self, G, personalization):
        if self.enabled:
            target_alpha = self.ranker.alpha
            target_tol = self.ranker.convergence.tol
            self.ranker.convergence.rank = None
            self.ranker.convergence.allow_restart = False
            alpha = target_alpha * 0.8
            beta = 0.5
            self.ranker.convergence.tol = 0.01
            while True:
                alpha = target_alpha * beta + alpha * (1 - beta)
                if abs(alpha - target_alpha) < 1 - target_alpha and alpha!=target_alpha:
                    alpha = target_alpha
                    self.ranker.convergence.tol = target_tol
                    self.ranker.convergence.force_next_iteration()
                print(self.ranker.convergence.iteration, alpha)
                ranks = self.ranker.rank(G, personalization, warm_start=self.ranker.convergence.rank)
                if alpha == target_alpha:
                    break
            self.ranker.convergence.allow_restart = True
        else:
            ranks = self.ranker.rank(G, personalization, warm_start=None)
        #print(self.ranker.convergence.elapsed_time, 'time,', self.ranker.convergence.iteration, 'iterations')
        return ranks