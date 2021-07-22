import scipy
import numpy as np
import warnings
import pygrank.algorithms.utils
from pygrank.algorithms.abstract_filters import RecursiveGraphFilter, ClosedFormGraphFilter


class PageRank(RecursiveGraphFilter):
    """A Personalized PageRank power method algorithm."""

    def __init__(self, alpha=0.85, *args, **kwargs):
        """ Initializes the PageRank scheme parameters.
        Attributes:
            alpha: Optional. 1-alpha is the bias towards the personalization. Default value is 0.85.
        Example:
            >>> from pygrank.algorithms import pagerank
            >>> algorithm = pagerank.PageRank(alpha=0.99, tol=1.E-9) # tol passed to the ConvergenceManager
        """
        self.alpha = float(alpha)
        super().__init__(*args, **kwargs)

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        #self.is_dangling = np.where(np.array(M.sum(axis=1)).flatten() == 0)[0]
        #ranks = self.alpha * (ranks * M + np.sum(ranks[is_dangling]) * personalization) + (1 - self.alpha) * personalization
        return self.alpha * (ranks * M) + (1 - self.alpha) * personalization


class HeatKernel(ClosedFormGraphFilter):
    """ Heat kernel filter."""

    def __init__(self, t=3, *args, **kwargs):
        """ Initializes the HearKernel filter parameters.
        Attributes:
            t: Optional. How many hops until the importance of new nodes starts decreasing. Default value is 5.
        Example:
            >>> from pygrank.algorithms import pagerank
            >>> algorithm = pagerank.HeatKernel(t=5, tol=1.E-9) # tol passed to the ConvergenceManager
        """
        self.t = float(t)
        super().__init__(*args, **kwargs)

    def _coefficient(self, previous_coefficient):
        return np.exp(-self.t) if previous_coefficient is None else (previous_coefficient*self.t/(self.convergence.iteration+1))


class AbsorbingRank(RecursiveGraphFilter):
    """ Implementation of partial absorbing random walks for Lambda = diag(absorbtion vector), e.g. Lambda = aI
    Wu, Xiao-Ming, et al. "Learning with partially absorbing random walks." Advances in neural information processing systems. 2012.
    """

    def __init__(self, alpha=1-1.E-6, to_scipy=None, use_quotient=True, convergence=None, converge_to_eigenvectors=False, **kwargs):
        """ Initializes the AbsorbingRank filter parameters.

        Attributes:
            alpha: Optional. (1-alpha)/alpha is the absorbtion rate of the random walk. This is chosen to yield the
                same underlying meaning as PageRank (for which Lambda = a Diag(degrees) )

        Example:
            >>> from pygrank.algorithms import pagerank
            >>> algorithm = pagerank.HeatKernel(t=5, tol=1.E-9) # tol passed to the ConvergenceManager
        """

        super().__init__(to_scipy=to_scipy, convergence=convergence, **kwargs)
        self.use_quotient = use_quotient
        self.converge_to_eigenvectors = converge_to_eigenvectors
        self.alpha = float(alpha) # typecast to make sure that a graph is not accidentally the first argument

    def _start(self, M, personalization, ranks, absorption=None, **kwargs):
        self.absorption = pygrank.algorithms.utils.to_signal(personalization, absorption).np * (1 - self.alpha) / self.alpha
        self.degrees = np.array(M.sum(axis=1)).flatten()

    def _end(self):
        del self.absorption
        del self.degrees

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        return ((ranks * M) * self.degrees + personalization * self.absorption) / (self.absorption + self.degrees)


class BiasedKernel(RecursiveGraphFilter):
    """ Heuristic kernel-like method that places emphasis on shorter random walks."""

    def __init__(self, alpha=0.85, t=1, *args, **kwargs):
        self.alpha = float(alpha)
        self.t = float(t)
        super().__init__(*args, **kwargs)
        warnings.warn("BiasedKernel is a low-quality heuristic", stacklevel=2)

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        a = self.alpha * self.t / self.convergence.iteration
        return personalization + a * ((ranks * M) * personalization) - ranks