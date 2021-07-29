import warnings
import pygrank.algorithms.utils
from pygrank import backend
from pygrank.algorithms.abstract_filters import RecursiveGraphFilter, ClosedFormGraphFilter


class PageRank(RecursiveGraphFilter):
    """A Personalized PageRank power method algorithm."""

    def __init__(self, alpha=0.85, *args, **kwargs):
        """ Initializes the PageRank scheme parameters.
        Args:
            alpha: Optional. 1-alpha is the bias towards the personalization. Default value is 0.85.
        Example:
            >>> from pygrank.algorithms import adhoc
            >>> algorithm = adhoc.PageRank(alpha=0.99, tol=1.E-9) # tol passed to the ConvergenceManager
        """
        self.alpha = float(alpha)
        super().__init__(*args, **kwargs)

    def _start(self, M, personalization, ranks, *args, **kwargs):
        super()._start(M, personalization, ranks, *args, **kwargs)
        #self.is_dangling = np.where(np.array(M.sum(axis=1)).flatten() == 0)[0]

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        #return self.alpha * (ranks * M + backend.sum(ranks[self.is_dangling]) * personalization) + (1 - self.alpha) * personalization
        return self.alpha * backend.conv(ranks, M) + (1 - self.alpha) * personalization

    def _end(self, M, personalization, ranks, *args, **kwargs):
        super()._end(M, personalization, ranks, *args, **kwargs)
        #del self.is_dangling


class HeatKernel(ClosedFormGraphFilter):
    """ Heat kernel filter."""

    def __init__(self, t=3, *args, **kwargs):
        """ Initializes the HearKernel filter parameters.

        Args:
            t: Optional. How many hops until the importance of new nodes starts decreasing. Default value is 5.

        Example:
            >>> from pygrank.algorithms import adhoc
            >>> algorithm = adhoc.HeatKernel(t=5, tol=1.E-9) # tol passed to the ConvergenceManager
        """
        self.t = float(t)
        super().__init__(*args, **kwargs)

    def _coefficient(self, previous_coefficient):
        return backend.exp(-self.t) if previous_coefficient is None else (previous_coefficient * self.t / (self.convergence.iteration + 1))


class AbsorbingWalks(RecursiveGraphFilter):
    """ Implementation of partial absorbing random walks for Lambda = (1-alpha)/alpha diag(absorbtion vector) .
    """

    def __init__(self, alpha=1-1.E-6, to_scipy=None, use_quotient=True, convergence=None, converge_to_eigenvectors=False, **kwargs):
        """ Initializes the AbsorbingWalks filter parameters. For appropriate parameter values. This can model PageRank
        but is in principle a generalization that allows custom absorbtion rates per nodes (when not given, these are I).

        Args:
            alpha: Optional. (1-alpha)/alpha is the absorbtion rate of the random walk multiplied with individual node
                absorbtion rates. This is chosen to yield the
                same underlying meaning as PageRank (for which Lambda = alpha Diag(degrees) ) when the same parameter value
                alpha is chosen. Default is 1-1.E-6 per the respective publication.

        Example:
            >>> from pygrank.algorithms import adhoc
            >>> algorithm = adhoc.AbsorbingWalks(0.85, tol=1.E-9) # tol passed to the ConvergenceManager
        """

        super().__init__(to_scipy=to_scipy, convergence=convergence, **kwargs)
        self.use_quotient = use_quotient
        self.converge_to_eigenvectors = converge_to_eigenvectors
        self.alpha = float(alpha) # typecast to make sure that a graph is not accidentally the first argument

    def _start(self, M, personalization, ranks, absorption=None, **kwargs):
        self.absorption = pygrank.algorithms.utils.to_signal(personalization, absorption).np * (1 - self.alpha) / self.alpha
        self.degrees = backend.degrees(M)

    def _end(self, *args, **kwargs):
        super()._end(*args, **kwargs)
        del self.absorption
        del self.degrees

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        return (backend.conv(ranks, M) * self.degrees + personalization * self.absorption) / (self.absorption + self.degrees)


class BiasedKernel(RecursiveGraphFilter):
    """ Heuristic kernel-like method that places emphasis on shorter random walks."""

    def __init__(self, alpha=0.85, t=1, *args, **kwargs):
        self.alpha = float(alpha)
        self.t = float(t)
        super().__init__(*args, **kwargs)
        warnings.warn("BiasedKernel is a low-quality heuristic", stacklevel=2)

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        a = self.alpha * self.t / self.convergence.iteration
        return personalization + a * (backend.conv(ranks, M) * personalization) - ranks
