from pygrank.algorithms.filters.abstract_filters import ClosedFormGraphFilter


class GenericGraphFilter(ClosedFormGraphFilter):
    """Implements a graph filter with a specific vector of weight parameters."""

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
        super(GenericGraphFilter, self).__init__(**kwargs)
        self.weights = weights if weights is not None else [0.9]*10

    def _coefficient(self, _):
        if self.convergence.iteration > len(self.weights):
            return 0
        return self.weights[self.convergence.iteration-1]
