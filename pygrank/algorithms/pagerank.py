import scipy
import numpy as np
import warnings
import pygrank.algorithms.utils


class PageRank:
    def __init__(self, alpha=0.85, normalization='auto', convergence_manager=None):
        self.alpha = float(alpha) # typecast to make sure that a graph is not accidentally the first argument
        self.normalization = normalization
        self.convergence = pygrank.algorithms.utils.ConvergenceManager() if convergence_manager is None else convergence_manager

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
    def __init__(self, t=5, normalization='auto', convergence_manager=None):
        self.t = t
        self.normalization = normalization
        self.convergence = pygrank.algorithms.utils.ConvergenceManager() if convergence_manager is None else convergence_manager

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
    def __init__(self, alpha=0.85, t=5, normalization='auto', convergence_manager=None):
        self.alpha = alpha
        self.normalization = normalization
        self.convergence = pygrank.algorithms.utils.ConvergenceManager() if convergence_manager is None else convergence_manager
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
