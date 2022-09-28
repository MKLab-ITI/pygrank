from pygrank.core import to_signal, NodeRanking, GraphSignalGraph, GraphSignalData, BackendGraph, BackendPrimitive, GraphSignal
from pygrank.core.utils import call, ensure_used_args
from pygrank.core.utils import preprocessor as default_preprocessor, obj2id
from pygrank.core import backend
from pygrank.algorithms.convergence import ConvergenceManager
from pygrank.algorithms.filters.krylov_space import krylov_base, krylov2original, krylov_error_bound
from pygrank.algorithms.postprocess import Postprocessor, Tautology
from typing import Union


class GraphFilter(NodeRanking):
    """Implements the base functionality of a graph filter that preprocesses a graph and an iterative computation scheme
    that stops based on a convergence manager."""

    def __init__(self,
                 preprocessor=None,
                 convergence=None,
                 personalization_transform: Postprocessor = None,
                 preserve_norm: bool = True,
                 ** kwargs):
        """
        Args:
            preprocessor: Optional. Method to extract a scipy sparse matrix from a networkx graph.
                If None (default), pygrank.algorithms.utils.preprocessor is used with keyword arguments
                automatically extracted from the ones passed to this constructor.
            convergence: Optional. The ConvergenceManager that determines when iterations stop. If None (default),
                a ConvergenceManager is used with keyword arguments
                automatically extracted from the ones passed to this constructor.
            personalization_transform: Optional. A Postprocessor whose `transform` method is used to transform
                the personalization before applying the graph filter. If None (default) a Tautology is used.
            preserve_norm: Optional. If True (default) the input's norm is used to scale the output. For example,
                if *convergence* is L1, this effectively means that the sum of output values is equal to the sum
                of input values.
        """
        self.preprocessor = call(default_preprocessor, kwargs) if preprocessor is None else preprocessor
        self.convergence = call(ConvergenceManager, kwargs) if convergence is None else convergence
        self.personalization_transform = Tautology() if personalization_transform is None else personalization_transform
        self.preserve_norm = preserve_norm
        ensure_used_args(kwargs, [default_preprocessor, ConvergenceManager])

    def _prepare(self, personalization: GraphSignal):
        pass

    def rank(self,
             graph: GraphSignalGraph = None,
             personalization: GraphSignalData = None,
             warm_start: GraphSignalData = None,
             graph_dropout: float = 0, *args, **kwargs) -> GraphSignal:
        personalization = to_signal(graph, personalization)
        self._prepare(personalization)
        personalization = self.personalization_transform(personalization)
        personalization_norm = backend.sum(backend.abs(personalization.np))
        if personalization_norm == 0:
            return personalization
        personalization = to_signal(personalization, personalization.np / personalization_norm)
        ranks = to_signal(personalization, backend.copy(personalization.np) if warm_start is None else warm_start)
        M = self.preprocessor(self._prepare_graph(personalization.graph, *args, **kwargs))
        self.convergence.start()
        self._start(backend.graph_dropout(M, graph_dropout), personalization, ranks, *args, **kwargs)
        while not self.convergence.has_converged(ranks.np):
            self._step(backend.graph_dropout(M, graph_dropout), personalization, ranks, *args, **kwargs)
        self._end(backend.graph_dropout(M, graph_dropout), personalization, ranks, *args, **kwargs)
        if self.preserve_norm:
            ranks.np = ranks.np * personalization_norm
        return ranks

    def _prepare_graph(self, graph, *args, **kwargs):
        return graph

    def _start(self, M, personalization, ranks, *args, **kwargs):
        pass

    def _end(self, M, personalization, ranks, *args, **kwargs):
        pass

    def _step(self, M, personalization, ranks, *args, **kwargs):
        raise Exception("Use a derived class of GraphFilter that implements the _step method")

    def references(self):
        return ["graph filter \\cite{ortega2018graph}"]

    def cite(self):
        ret = super().cite()
        if isinstance(self.personalization_transform, Tautology) and self.personalization_transform.ranker is None:
            return ret
        return self.personalization_transform.cite() + "\n  passed to " + ret

    def __add__(self, other):
        if isinstance(other, ConvergenceManager):
            self.convergence = other
        elif hasattr(other, "__name__") and other.__name__ == "preprocess":
            self.preprocessor = other
        elif isinstance(other, Postprocessor):
            self.use_quotient = other
        else:
            raise Exception("Can only add convergence managers and preprocessors to graph filters")
        return self

    def __lshift__(self, ranker):
        if not isinstance(ranker, NodeRanking):
            raise Exception("pygrank can only shift rankers into filters")
        self.personalization_transform = ranker
        return ranker


class RecursiveGraphFilter(GraphFilter):
    """Implements a graph filter described through a recursion ranks = formula(G, ranks)"""

    def __init__(self,
                 use_quotient: Union[bool, Postprocessor] = True,
                 converge_to_eigenvectors: bool = False,
                 *args, **kwargs):
        """
        Args:
            use_quotient: Optional. If True (default) performs a L1 re-normalization of ranks after each iteration.
                This significantly speeds up the convergence speed of symmetric normalization (col normalization
                preserves the L1 norm during computations on its own). Provide `pygrank.Postprocessor` or other
                callable instances to adjust node scores after each iteration.
                Can pass False or None to ignore this functionality and make recursive filter outcome equal to
                its expansion.
        """
        super().__init__(*args, **kwargs)
        self.use_quotient = use_quotient
        self.converge_to_eigenvectors = converge_to_eigenvectors

    def _step(self, M, personalization, ranks, *args, **kwargs):
        ranks.np = self._formula(M, personalization, ranks, *args, **kwargs)
        if isinstance(ranks.np, GraphSignal):
            ranks.np = ranks.np.np

        if isinstance(self.use_quotient, Postprocessor):
            ranks.np = self.use_quotient(ranks)
        elif self.use_quotient:
            ranks.np = backend.safe_div(ranks, backend.sum(ranks))
        if self.converge_to_eigenvectors:
            personalization.np = ranks.np

    def _formula(self,
                 M: BackendGraph,
                 personalization: BackendPrimitive,
                 ranks: BackendPrimitive,
                 *args, **kwargs):
        raise Exception("Use a derived class of RecursiveGraphFilter that implements the _formula method")

    def references(self):
        refs = super().references()
        if self.converge_to_eigenvectors:
            refs += ["unbiased eigenvector convergence \\cite{krasanakis2018venuerank}"]
        return refs


class ClosedFormGraphFilter(GraphFilter):
    """Implements a graph filter described as an aggregation of graph signal diffusion certain number of hops away
    while weighting these by corresponding coefficients."""

    def __init__(self,
                 krylov_dims: int = None,
                 coefficient_type: str = "taylor",
                 optimization_dict: dict = None,
                 *args, **kwargs):
        """
        Args:
            krylov_dims: Optional. Performs the Lanczos method to estimate filter outcome in the Krylov space
                of the graph with degree equal to the provided dimensions. This considerably speeds up filtering
                but ends up providing an *approximation* of true graph filter outcomes.
                If None (default) filters are computed in the node space, which can be slower but
                yields exact computations. Otherwise, a numeric value
                equal to the number of latent Krylov space dimensions is required.
            coefficient_type: Optional. If "taylor" (default), provided coefficients are considered
                to define a Taylor expansion. If "chebyshev", they are considered to be the coefficients of a Chebyshev
                expansion, which provides more robust errors but require normalized personalization. These approaches
                are **not equivalent** for the same coefficient values; changing this argument could cause adhoc
                filters to not work as indented.
            optimization_dict: Optional. If it is a dict, the filter keeps intermediate values that can help it
                avoid most (if not all) matrix multiplication when run again for the same graph signal. Setting this
                parameter to None (default) can save approximately **half the memory** the algorithm uses but
                slows down tuning iteration times to O(edges) instead of O(nodes). Note that the same dict needs to
                be potentially passed to multiple algorithms that take the same graph signal as input to see any
                improvement.
        """
        super().__init__(*args, **kwargs)
        self.krylov_dims = krylov_dims
        self.coefficient_type = coefficient_type.lower()
        self.optimization_dict = optimization_dict

    def references(self):
        refs = super().references()
        if self.coefficient_type == "chebyshev":
            refs.append("Chebyshev coefficients \\cite{yu2021chebyshev}")
        if self.krylov_dims is not None:
            refs.append("Lanczos acceleration \\cite{susnjara2015accelerated} in the"+str(self.krylov_dims)+"-dimensional Krylov space")
        if self.optimization_dict is not None:
            refs.append("dictionary-based hashing \\cite{krasanakis2021pygrank}")
        return refs

    def _start(self, M, personalization, ranks, *args, **kwargs):
        self.coefficient = None
        if self.coefficient_type == "chebyshev":
            self.prev_term = 0
        if self.krylov_dims is not None:
            V, H = krylov_base(M, personalization.np, int(self.krylov_dims))
            self.krylov_base = V
            self.krylov_H = H
            self.zero_coefficient = self.coefficient
            self.krylov_result = H*0
            self.Mpower = backend.eye(int(self.krylov_dims))
            error_bound = krylov_error_bound(V, H, M, personalization.np)
            if error_bound > 0.01:
                raise Exception("Krylov approximation with estimated relative error "+str(error_bound)
                              + " > 0.01 is too rough to be meaningful (try on lager graphs)")
        else:
            self.ranks_power = personalization.np
            ranks.np = backend.repeat(0.0, backend.length(ranks.np))

    def _recursion(self, result, next_term, next_coefficient):
        if self.coefficient_type == "chebyshev":
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

    def _prepare(self, personalization: GraphSignal):
        if self.optimization_dict is not None:
            personalization_id = obj2id(personalization)
            if personalization_id not in self.optimization_dict:
                self.optimization_dict[personalization_id] = dict()
            self.__active_dict = self.optimization_dict[personalization_id]
        else:
            self.__active_dict = None

    def _retrieve_power(self, ranks_power, M):
        if self.__active_dict is not None:
            if self.convergence.iteration not in self.__active_dict:
                self.__active_dict[self.convergence.iteration] = backend.conv(ranks_power, M) if self.krylov_dims is None else ranks_power @ M
            return self.__active_dict[self.convergence.iteration]
        return backend.conv(ranks_power, M) if self.krylov_dims is None else ranks_power @ M

    def _step(self, M, personalization, ranks, *args, **kwargs):
        self.coefficient = self._coefficient(self.coefficient)
        if self.krylov_dims is not None:
            self.krylov_result, self.Mpower = self._recursion(self.krylov_result, self.Mpower, self.coefficient)
            ranks.np = krylov2original(self.krylov_base, self.krylov_result, int(self.krylov_dims))
            self.Mpower = self._retrieve_power(self.Mpower, self.krylov_H)
        else:
            ranks.np, self.ranks_power = self._recursion(ranks.np, self.ranks_power, self.coefficient)
            self.ranks_power = self._retrieve_power(self.ranks_power, M)

    def _end(self, M, personalization, ranks, *args, **kwargs):
        if self.krylov_dims is not None:
            del self.krylov_base
            del self.krylov_H
        else:
            del self.ranks_power
        if self.coefficient_type == "chebyshev":
            del self.prev_term
        del self.coefficient
        del self.__active_dict

    def _coefficient(self, previous_coefficient: float) -> float:
        raise Exception("Use a derived class of ClosedFormGraphFilter that implements the _coefficient method")
