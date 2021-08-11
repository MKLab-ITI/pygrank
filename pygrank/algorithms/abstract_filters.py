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
        ranks = to_signal(personalization, backend.copy(personalization.np) if warm_start is None else warm_start)
        M = self.to_scipy(personalization.graph)
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

    def __init__(self, krylov_dims=None, coefficient_type="taylor", *args, **kwargs):
        """
        Args:
            krylov_dims: Optional. Performs the Lanczos method to estimate filter outcome in the Krylov space
                of the graph with degree equal to the provided dimensions. This considerably speeds up filtering
                but ends up providing an *approximation* of true graph filter outcomes.
                If None (default) filters are not computed through their projection
                the Krylov space, which may yield slower but exact computations. Otherwise, a numeric value
                equal to the number of latent dimensions is required.
            coefficient_type: Optional. If "taylor" (default) provided coefficients are considered
                to define a Taylor expansion. If "chebychev", they are considered to be the coefficients of a Chebychev
                expansion, which provides more robust errors but require normalized personalization. These approaches
                are **not equivalent** for the same coefficient values; changing this argument could cause adhoc
                filters to not work as indented.
        """
        super().__init__(*args, **kwargs)
        self.krylov_dims = krylov_dims
        self.coefficient_type = coefficient_type.lower()

    def _start(self, M, personalization, ranks, *args, **kwargs):
        self.coefficient = None
        if self.coefficient_type == "chebychev":
            self.prev_term = 0
        if self.krylov_dims is not None:
            V, H = krylov_base(M, personalization.np, int(self.krylov_dims))
            self.krylov_base = V
            self.krylov_H = H
            self.zero_coefficient = self.coefficient
            self.krylov_result = 0
            self.Mpower = backend.eye(int(self.krylov_dims))
        else:
            self.ranks_power = personalization.np
            ranks.np = backend.repeat(0.0, backend.length(ranks.np))

    def _recursion(self, result, next_term, next_coefficient):
        if self.coefficient_type == "chebychev":
            if self.convergence.iteration == 2:
                self.prev_term = next_term
            if self.convergence.iteration > 2:
                next_term = 2*next_term - self.prev_term
                self.prev_term = next_term
                if self.coefficient == 0:
                    return result, next_term
            return result + next_term*next_coefficient, next_term
        elif self.coefficient_type == "taylor":
            if self.coefficient == 0:
                return result, next_term
            return result + next_term*next_coefficient, next_term
        else:
            raise Exception("Invalid coefficient type")

    def _step(self, M, personalization, ranks, *args, **kwargs):
        self.coefficient = self._coefficient(self.coefficient)
        if self.krylov_dims is not None:
            prevPower = self.Mpower
            self.Mpower = self.Mpower @ self.krylov_H
            self.krylov_result, self.Mpower = self._recursion(self.krylov_result, self.Mpower, self.coefficient)
            #self.krylov_result += self.coefficient * self.Mpower
            ranks.np = krylov2original(self.krylov_base, self.krylov_result, int(self.krylov_dims))
        else:
            #if self.coefficient != 0:
            #    ranks.np = ranks.np + float(self.coefficient) * self.ranks_power
            ranks.np, self.ranks_power = self._recursion(ranks.np, self.ranks_power, float(self.coefficient))
            self.ranks_power = backend.conv(self.ranks_power, M)

    def _end(self, M, personalization, ranks, *args, **kwargs):
        if self.krylov_dims is not None:
            #if self.convergence.iteration >= int(self.krylov_dims):
            #    warnings.warn("Robust Krylov space approximations require at least one degree higher than the number of coefficients.\n"
            #                  +"Consider setting krylov_dims="+str(self.convergence.iteration+1)+" or more in the constructor.", stacklevel=2)
            del self.krylov_base
            del self.krylov_H
        else:
            del self.ranks_power
        if self.coefficient_type == "chebychev":
            del self.prev_term
        del self.coefficient

    def _coefficient(self, previous_coefficient):
        raise Exception("Use a derived class of ClosedFormGraphFilter that implements the _coefficient method")
