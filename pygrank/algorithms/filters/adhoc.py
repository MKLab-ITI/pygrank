import warnings
from pygrank.core.signals import to_signal
from pygrank import backend
from pygrank.algorithms.filters.abstract_filters import RecursiveGraphFilter, ClosedFormGraphFilter


class PageRank(RecursiveGraphFilter):
    """A Personalized PageRank power method algorithm."""

    def __init__(self, alpha=0.85, *args, **kwargs):
        """ Initializes the PageRank scheme parameters.
        Args:
            alpha: Optional. 1-alpha is the bias towards the personalization. Default value is 0.85.
        Example:
            >>> from pygrank.algorithms import PageRank
            >>> algorithm = PageRank(alpha=0.99, tol=1.E-9) # tol passed to the ConvergenceManager
        """
        self.alpha = float(alpha)
        super().__init__(*args, **kwargs)

    def _start(self, M, personalization, ranks, *args, **kwargs):
        # TODO: self.is_dangling = np.where(np.array(M.sum(axis=1)).flatten() == 0)[0]
        super()._start(M, personalization, ranks, *args, **kwargs)

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        # TODO: return self.alpha * (ranks * M + backend.sum(ranks[self.is_dangling]) * personalization) + (1 - self.alpha) * personalization
        return self.alpha * backend.conv(ranks, M) + (1 - self.alpha) * personalization

    def _end(self, M, personalization, ranks, *args, **kwargs):
        # TODO: del self.is_dangling
        super()._end(M, personalization, ranks, *args, **kwargs)


class HeatKernel(ClosedFormGraphFilter):
    """ Heat kernel filter."""

    def __init__(self, t=3, *args, **kwargs):
        """ Initializes the HeatKernel filter parameters.

        Args:
            t: Optional. How many hops until the importance of new nodes starts decreasing. Default value is 5.

        Example:
            >>> from pygrank.algorithms import HeatKernel
            >>> algorithm = HeatKernel(t=3, tol=1.E-9) # tol passed to the ConvergenceManager
        """
        self.t = float(t)
        super().__init__(*args, **kwargs)

    def _coefficient(self, previous_coefficient):
        return backend.exp(-self.t) if previous_coefficient is None else (previous_coefficient * self.t / (self.convergence.iteration + 1))


class AbsorbingWalks(RecursiveGraphFilter):
    """ Implementation of partial absorbing random walks for Lambda = (1-alpha)/alpha diag(absorption vector) .
    """

    def __init__(self, alpha=1-1.E-6, *args, **kwargs):
        """ Initializes the AbsorbingWalks filter parameters. For appropriate parameter values. This can model PageRank
        but is in principle a generalization that allows custom absorption rate per node (when not given, these are I).

        Args:
            alpha: Optional. (1-alpha)/alpha is the absorption rate of the random walk multiplied with individual node
                absorption rates. This is chosen to yield the
                same underlying meaning as PageRank (for which Lambda = alpha Diag(degrees) ) when the same parameter
                value alpha is chosen. Default is 1-1.E-6 per the respective publication.

        Example:
            >>> from pygrank.algorithms import AbsorbingWalks
            >>> algorithm = AbsorbingWalks(0.85, tol=1.E-9) # tol passed to the ConvergenceManager
        """

        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)  # typecast to make sure that a graph is not accidentally the first argument

    def _start(self, M, personalization, ranks, absorption=None, **kwargs):
        self.absorption = to_signal(personalization.graph, absorption).np * (1 - self.alpha) / self.alpha
        self.degrees = backend.degrees(M)

    def _end(self, *args, **kwargs):
        super()._end(*args, **kwargs)
        del self.absorption
        del self.degrees

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        ret = (backend.conv(ranks, M) * self.degrees + personalization * self.absorption) / (self.absorption + self.degrees)
        return ret


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
