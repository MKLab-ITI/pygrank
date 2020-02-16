import scipy
import numpy as np
import warnings
import pygrank.algorithms.utils


class PageRank:
    """A Personalized PageRank power method algorithm. Supports warm start."""

    def __init__(self, alpha=0.85, to_scipy=None, convergence=None, use_quotient=True, **kwargs):
        """ Initializes the PageRank scheme parameters.

        Attributes:
            alpha: Optional. 1-alpha is the bias towards the personalization. Default value is 0.85.
            to_scipy: Optional. Method to extract a scipy sparse matrix from a networkx graph.
                If None (default), pygrank.algorithms.utils.to_scipy_sparse_matrix with default arguments is used.
            convergence: Optional. The ConvergenceManager that determines when iterations stop. If None (default),
                a ConvergenceManager with the additional keyword arguments is constructed.
            use_quotient: Optional. If True (default) performs a L1 re-normalization of ranks after each iteration.
                This significantly speeds ups the convergence speed of symmetric normalization (col normalization
                preserves the L1 norm during computations on its own).

        Example:
            >>> from pygrank.algorithms import pagerank
            >>> algorithm = pagerank.PageRank(alpha=0.99, tol=1.E-9) # tol passed to the ConvergenceManager
        """
        self.alpha = float(alpha) # typecast to make sure that a graph is not accidentally the first argument
        self.to_scipy = pygrank.algorithms.utils.to_scipy_sparse_matrix if to_scipy is None else to_scipy
        self.convergence = pygrank.algorithms.utils.ConvergenceManager(**kwargs) if convergence is None else convergence
        self.use_quotient = use_quotient

    def rank(self, G, personalization=None, warm_start=None):
        M = self.to_scipy(G)
        degrees = scipy.array(M.sum(axis=1)).flatten()

        personalization = scipy.repeat(1.0, len(G)) if personalization is None else scipy.array([personalization.get(n, 0) for n in G], dtype=float)
        if personalization.sum() == 0:
            raise Exception("The personalization vector should contain at least one non-zero entity")
        personalization = personalization / personalization.sum()
        ranks = personalization if warm_start is None else scipy.array([warm_start.get(n, 0) for n in G], dtype=float)

        is_dangling = scipy.where(degrees == 0)[0]
        self.convergence.start()
        while not self.convergence.has_converged(ranks):
            ranks = self.alpha * (ranks * M + sum(ranks[is_dangling]) * personalization) + (1 - self.alpha) * personalization
            if self.use_quotient:
                ranks = ranks/ranks.sum()

        ranks = dict(zip(G.nodes(), map(float, ranks)))
        return ranks


class HeatKernel:
    """ Heat kernel filter."""

    def __init__(self, t=5, to_scipy=None, convergence=None, **kwargs):
        """ Initializes the HearKernel filter parameters.

        Attributes:
            t: Optional. How many hops until the importance of new nodes starts decreasing. Default value is 5.
            to_scipy: Optional. Method to extract a scipy sparse matrix from a networkx graph.
                If None (default), pygrank.algorithms.utils.to_scipy_sparse_matrix with default arguments is used.
            convergence: Optional. The ConvergenceManager that determines when iterations stop. If None (default),
                a ConvergenceManager with the additional keyword arguments is constructed.

        Example:
            >>> from pygrank.algorithms import pagerank
            >>> algorithm = pagerank.HeatKernel(t=5, tol=1.E-9) # tol passed to the ConvergenceManager
        """
        self.t = t
        self.to_scipy = pygrank.algorithms.utils.to_scipy_sparse_matrix if to_scipy is None else to_scipy
        self.convergence = pygrank.algorithms.utils.ConvergenceManager(**kwargs) if convergence is None else convergence

    def rank(self, G, personalization=None):
        M = self.to_scipy(G)

        personalization = scipy.repeat(1.0, len(G)) if personalization is None else scipy.array([personalization.get(n, 0) for n in G], dtype=float)
        if personalization.sum() == 0:
            raise Exception("The personalization vector should contain at least one non-zero entity")
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

    def __init__(self, alpha=0.85, t=5, to_scipy=None, convergence=None, **kwargs):
        self.alpha = alpha
        self.t = t
        self.to_scipy = pygrank.algorithms.utils.to_scipy_sparse_matrix if to_scipy is None else to_scipy
        self.convergence = pygrank.algorithms.utils.ConvergenceManager(**kwargs) if convergence is None else convergence
        warnings.warn("BiasedKernel is still under development (its implementation may be incorrect)", stacklevel=2)
        warnings.warn("BiasedKernel is a low-quality heuristic", stacklevel=2)

    def rank(self, G, personalization=None, warm_start=None):
        M = self.to_scipy(G)
        degrees = scipy.array(M.sum(axis=1)).flatten()

        personalization = scipy.repeat(1.0, len(G)) if personalization is None else scipy.array([personalization.get(n, 0) for n in G], dtype=float)
        if personalization.sum() == 0:
            raise Exception("The personalization vector should contain at least one non-zero entity")
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
