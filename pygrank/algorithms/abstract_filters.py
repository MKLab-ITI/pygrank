from .utils import _call, _ensure_all_used
from pygrank.algorithms.utils import NodeRanking, preprocessor, ConvergenceManager, to_signal, krylov_base, krylov2original
from pygrank import backend
from pygrank.algorithms.postprocess import Postprocessor


class GraphFilter(NodeRanking):
    """Implements the base functionality of a graph filter that preprocesses a graph and an iterative computation scheme
    that stops based on a convergence manager."""

    def __init__(self, to_scipy = None, convergence = None, ** kwargs):
        """
        Args:
            to_scipy: Optional. Method to extract a scipy sparse matrix from a networkx graph.
                If None (default), pygrank.algorithms.utils.preprocessor is used with keyword arguments
                automatically extracted from the ones passed to this constructor.
            convergence: Optional. The ConvergenceManager that determines when iterations stop. If None (default),
                a ConvergenceManager is used with keyword arguments
                automatically extracted from the ones passed to this constructor.
        """
        self.to_scipy = _call(preprocessor, kwargs) if to_scipy is None else to_scipy
        self.convergence = _call(ConvergenceManager, kwargs) if convergence is None else convergence
        _ensure_all_used(kwargs, [preprocessor, ConvergenceManager])

    def rank(self, graph=None, personalization=None, warm_start=None, normalize_personalization=True, *args, **kwargs):
        personalization = to_signal(graph, personalization).normalized(normalize_personalization)
        if backend.sum(backend.abs(personalization.np)) == 0:
            raise Exception("Personalization should contain at least one non-zero entity")
        if graph is None:
            graph = personalization.G
        ranks = to_signal(graph, backend.copy(personalization.np) if warm_start is None else warm_start)
        M = self.to_scipy(graph)
        self.convergence.start()
        self._start(M, personalization, ranks, *args, **kwargs)
        while not self.convergence.has_converged(ranks.np):
            self._step(M, personalization, ranks, *args, **kwargs)
        self._end(M, personalization, ranks, *args, **kwargs)
        return ranks

    def _start(self, M, personalization, ranks, *args, **kwargs):
        pass

    def _end(self, M, personalization, ranks, *args, **kwargs):
        pass

    def _step(self, M, personalization, ranks, *args, **kwargs):
        raise Exception("Use a derived class of GraphFilter that implements the _step method")


class RecursiveGraphFilter(GraphFilter):
    """Implements a graph filter described through recursion ranks = formula(G, ranks)"""

    def __init__(self, use_quotient=True, converge_to_eigenvectors=False, *args, **kwargs):
        """
        Args:
            use_quotient: Optional. If True (default) performs a L1 re-normalization of ranks after each iteration.
                This significantly speeds up the convergence speed of symmetric normalization (col normalization
                preserves the L1 norm during computations on its own). Can also pass Postprocessor instances
                to adjust node scores after each iteration with the Postprocessor.transform(ranks) method.
                Can pass False or None to ignore this parameter's functionality.
        """
        super().__init__(*args, **kwargs)
        self.use_quotient = use_quotient
        self.converge_to_eigenvectors = converge_to_eigenvectors

    def _step(self, M, personalization, ranks, *args, **kwargs):
        ranks.np = self._formula(M, personalization.np, ranks.np, *args, **kwargs)
        if isinstance(self.use_quotient, Postprocessor):
            ranks.np = self.use_quotient.transform(ranks).np
        elif self.use_quotient:
            ranks.np = ranks.np / backend.sum(ranks.np)
        if self.converge_to_eigenvectors:
            personalization.np = ranks.np

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        raise Exception("Use a derived class of RecursiveGraphFilter that implements the _formula method")


class ClosedFormGraphFilter(GraphFilter):
    """Implements a graph filter described as an aggregation of graph signal diffusion certain number of hops away
    while weighting these by corresponding coefficients."""

    def __init__(self, krylov_dims=None, *args, **kwargs):
        """
        Args:
            krylov_dims: Optional. Performs the Lanczos method to estimate filter outcome in the Krylov space
                of the graph with degree equal to the provided dimensions. This considerably speeds up filtering
                but ends up providing an *approximation* of true graph filter outcomes.
                If None (default) filters are not computed through their projection
                the Krylov space, which may yield slower but exact computations. Otherwise, a numeric value
                equal to the number of latent dimensions is required.
        """
        super().__init__(*args, **kwargs)
        self.krylov_dims = krylov_dims

    def _start(self, M, personalization, ranks, *args, **kwargs):
        self.coefficient = None
        if self.krylov_dims is not None:
            V, H = krylov_base(M, personalization.np, int(self.krylov_dims))
            self.krylov_base = V
            self.krylov_H = H
            self.zero_coefficient = self.coefficient
            self.krylov_result = 0
            self.Mpower = backend.eye(int(self.krylov_dims))
        else:
            self.ranks_power = ranks.np
            ranks.np = 0

    def _step(self, M, personalization, ranks, *args, **kwargs):
        self.coefficient = self._coefficient(self.coefficient)
        if self.coefficient == 0:
            return
        if self.krylov_dims is not None:
            self.Mpower = self.Mpower @ self.krylov_H
            self.krylov_result += self.coefficient * self.Mpower
            ranks.np = krylov2original(self.krylov_base, self.krylov_result, int(self.krylov_dims))
        else:
            self.ranks_power = backend.conv(self.ranks_power, M)
            ranks.np = ranks.np + float(self.coefficient) * self.ranks_power

    def _end(self, M, personalization, ranks, *args, **kwargs):
        if self.krylov_dims is not None:
            #if self.convergence.iteration >= int(self.krylov_dims):
            #    warnings.warn("Robust Krylov space approximations require at least one degree higher than the number of coefficients.\n"
            #                  +"Consider setting krylov_dims="+str(self.convergence.iteration+1)+" or more in the constructor.", stacklevel=2)
            del self.krylov_base
            del self.krylov_H
        del self.ranks_power
        del self.coefficient

    def _coefficient(self, previous_coefficient):
        raise Exception("Use a derived class of ClosedFormGraphFilter that implements the _coefficient method")
