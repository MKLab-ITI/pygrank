from .utils import _call, _ensure_all_used
import pygrank.algorithms.utils
import numpy as np


class GraphFilter(object):
    """Implements the base functionality of a graph filter that preprocesses a graph and an iterative computation scheme
    that stops based on a convergence manager."""

    def __init__(self, to_scipy = None, convergence = None, ** kwargs):
        """
        Args:
            to_scipy: Optional. Method to extract a scipy sparse matrix from a networkx graph.
                If None (default), pygrank.algorithms.utils.preprocessor is used with keyword arguments
                automatically extracted from the ones passed to this constructor.
            convergence: Optional. The ConvergenceManager that determines when iterations stop. If None (default),
                a ConvergenceManager with keyword arguments
                automatically extracted from the ones passed to this constructor.
        """
        self.to_scipy = _call(pygrank.algorithms.utils.preprocessor, kwargs) if to_scipy is None else to_scipy
        self.convergence = _call(pygrank.algorithms.utils.ConvergenceManager, kwargs) if convergence is None else convergence
        _ensure_all_used(kwargs, [pygrank.algorithms.utils.preprocessor, pygrank.algorithms.utils.vectorize, pygrank.algorithms.utils.ConvergenceManager])

    def rank(self, graph=None, personalization=None, warm_start=None, normalized_personalization=True, *args, **kwargs):
        self.convergence.start()
        personalization = pygrank.algorithms.utils.to_signal(graph, personalization).normalized(normalized_personalization)
        if np.abs(personalization.np).sum() == 0:
            raise Exception("Personalization should contain at least one non-zero entity")
        if graph is None:
            graph = personalization.G
        ranks = pygrank.algorithms.utils.to_signal(graph, personalization.np if warm_start is None else warm_start)
        M = self.to_scipy(graph)
        self._start(M, personalization, ranks, *args, **kwargs)
        while not self.convergence.has_converged(ranks.np):
            self._step(M, personalization, ranks, *args, **kwargs)
        self._end()
        return ranks

    def _start(self, M, personalization, ranks, *args, **kwargs):
        pass

    def _end(self):
        pass

    def _step(self, M, personalization, ranks, *args, **kwargs):
        raise Exception("Use a derived class of GraphFilter that implements the _step method")


class RecursiveGraphFilter(GraphFilter):
    """Implements a graph filter described through recursion ranks = formula(G, ranks)"""

    def __init__(self, use_quotient=True, converge_to_eigenvectors=False, *args, **kwargs):
        """
        Args:
            use_quotient: Optional. If True (default) performs a L1 re-normalization of ranks after each iteration.
                This significantly speeds ups the convergence speed of symmetric normalization (col normalization
                preserves the L1 norm during computations on its own). Can also pass a pygrank.algorithm.postprocess
                filter to perform any kind of normalization through its postprocess method. Note that these can slow
                down computations due to needing to convert ranks between skipy and maps after each iteration.
                Can pass False or None to ignore this parameter's functionality.
        """
        super().__init__(*args, **kwargs)
        self.use_quotient = use_quotient
        self.converge_to_eigenvectors = converge_to_eigenvectors

    def _step(self, M, personalization, ranks, *args, **kwargs):
        ranks.np = self._formula(M, personalization.np, ranks.np, *args, **kwargs)
        if self.use_quotient:
            ranks.np = ranks.np / ranks.np.sum()
        if self.converge_to_eigenvectors:
            personalization.np = ranks.np

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        raise Exception("Use a derived class of RecursiveGraphFilter that implements the _formula method")


class ClosedFormGraphFilter(GraphFilter):
    """Implements a graph filter described as an aggregation of graph signal diffusion certain number of hops away
    while weighting these by corresponding coefficients."""

    def _start(self, M, personalization, ranks, *args, **kwargs):
        self.Mpower = 1
        self.coefficient = self._coefficient(None)
        ranks.np *= self.coefficient

    def _step(self, M, personalization, ranks, *args, **kwargs):
        self.Mpower *= M
        self.coefficient = self._coefficient(self.coefficient)
        ranks.np += self.coefficient * self.Mpower * personalization.np

    def _end(self):
        del self.Mpower
        del self.coefficient

    def _coefficient(self, previous_coefficient):
        raise Exception("Use a derived class of ClosedFormGraphFilter that implements the _coefficient method")
