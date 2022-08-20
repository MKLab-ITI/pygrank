from pygrank.algorithms.filters import ClosedFormGraphFilter, GraphFilter
from pygrank.core import backend


class GenericGraphFilter(ClosedFormGraphFilter):
    """Defines a graph filter via its hop weight parameters."""

    def __init__(self, weights=None, **kwargs):
        """
        Initializes the graph filter.

        Args:
            weights: Optional. A list-like object with elements weights[n] proportional to the importance of propagating
                personalization graph signals n hops away. If None (default) then [0.9]*10 is used.

        Example:
            >>> from pygrank import GenericGraphFilter
            >>> algorithm = GenericGraphFilter([0.5, 0.25, 0.125], tol=1.E-9) # tol passed to ConvergenceManager
        """
        super().__init__(**kwargs)
        self.weights = weights if weights is not None else [0.9]*10

    def _coefficient(self, _):
        if self.convergence.iteration > len(self.weights):
            return 0
        return self.weights[self.convergence.iteration-1]


class ImpulseGraphFilter(GraphFilter):
    """Defines a graph filter with a specific vector of impulse response parameters."""

    def __init__(self, params=None, *args, **kwargs):
        """
        Initializes the graph filter.

        Args:
            params: Optional. A list-like object with elements weights[n] proportional to the impulse response
                when propagating graph signals at hop n. If None (default) then [0.9]*10 is used.

        Example:
            >>> from pygrank import GenericGraphFilter
            >>> algorithm = ImpulseGraphFilter([0.5, 0.5, 0.5], tol=None)  # tol=None runs all iterations
        """
        if params is None:
            params = [0.9]*10
        super().__init__(*args, **kwargs)
        self.params = params

    def _step(self, M, personalization, ranks, *args, **kwargs):
        if self.convergence.iteration > len(self.params):
            return 0
        param = self.params[self.convergence.iteration-1]
        if param == 0:
            return ranks
        if param == 1:
            ranks.np = backend.conv(ranks, M).np
            return ranks
        ranks.np = (backend.conv(ranks, M)*param + ranks*(1-param)).np


class LowPassRecursiveGraphFilter(GraphFilter):
    """Defines a low-pass graph filter with specific yet changing recursive terms."""

    def __init__(self, params=None, *args, **kwargs):
        """
        Initializes the graph filter.

        Args:
            params: Optional. A list-like object with elements weights[n] proportional to the impulse response
                when propagating graph signals at hop n. If None (default) then [0.9]*10 is used. This is
                equivalent to pygrank.PageRank(0.9, use_quotient=False, max_iters=10)

        Example:
            >>> from pygrank import LowPassRecursiveGraphFilter
            >>> algorithm = LowPassRecursiveGraphFilter([0.9]*10, tol=None)  # tol=None runs all iterations
        """
        if params is None:
            params = [0.9]*10
        super().__init__(*args, **kwargs)
        self.params = params

    def _step(self, M, personalization, ranks, *args, **kwargs):
        if self.convergence.iteration > len(self.params):
            return 0
        param = self.params[self.convergence.iteration-1]
        if param == 0:
            return ranks
        if param == 1:
            ranks.np = backend.conv(ranks, M).np
            return ranks
        ranks.np = (backend.conv(ranks, M)*param + personalization*(1-param)).np
