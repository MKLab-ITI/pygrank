from pygrank.core import (
    to_signal,
    NodeRanking,
    GraphSignalGraph,
    GraphSignalData,
    GraphSignal,
)
from pygrank.core.utils import call, ensure_used_args
from pygrank.core.utils import preprocessor as default_preprocessor
from pygrank.core import backend
from pygrank.algorithms.convergence import ConvergenceManager
from pygrank.algorithms.postprocess import Postprocessor, Tautology


class GraphFilter(NodeRanking):
    """Implements the base functionality of a graph filter that preprocesses a graph and an iterative computation scheme
    that stops based on a convergence manager."""

    def __init__(
        self,
        preprocessor=None,
        convergence=None,
        personalization_transform: Postprocessor = None,
        preserve_norm: bool = True,
        **kwargs
    ):
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
        self.preprocessor = (
            call(default_preprocessor, kwargs) if preprocessor is None else preprocessor
        )
        self.convergence = (
            call(ConvergenceManager, kwargs) if convergence is None else convergence
        )
        self.personalization_transform = (
            Tautology()
            if personalization_transform is None
            else personalization_transform
        )
        self.preserve_norm = preserve_norm
        ensure_used_args(kwargs, [default_preprocessor, ConvergenceManager])

    def _prepare(self, personalization: GraphSignal):
        pass

    def rank(
        self,
        graph: GraphSignalGraph = None,
        personalization: GraphSignalData = None,
        warm_start: GraphSignalData = None,
        graph_dropout: float = 0,
        *args,
        **kwargs
    ) -> GraphSignal:
        personalization = to_signal(graph, personalization)
        self._prepare(personalization)
        personalization = self.personalization_transform(personalization)
        personalization_norm = backend.sum(backend.abs(personalization.np))
        if personalization_norm == 0:
            return personalization
        personalization = to_signal(
            personalization, personalization.np / personalization_norm
        )
        ranks = to_signal(
            personalization,
            backend.copy(personalization.np) if warm_start is None else warm_start,
        )
        M = self.preprocessor(
            self._prepare_graph(personalization.graph, personalization, *args, **kwargs)
        )
        self.convergence.start()
        self._start(
            backend.graph_dropout(M, graph_dropout),
            personalization,
            ranks,
            *args,
            **kwargs
        )
        while not self.convergence.has_converged(ranks.np):
            self._step(
                backend.graph_dropout(M, graph_dropout),
                personalization,
                ranks,
                *args,
                **kwargs
            )
        self._end(
            backend.graph_dropout(M, graph_dropout),
            personalization,
            ranks,
            *args,
            **kwargs
        )
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
        raise Exception(
            "Use a derived class of GraphFilter that implements the _step method"
        )

    def references(self):
        return ["graph filter \\cite{ortega2018graph}"]

    def cite(self):
        ret = super().cite()
        if (
            isinstance(self.personalization_transform, Tautology)
            and self.personalization_transform.ranker is None
        ):
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
            raise Exception(
                "Can only add convergence managers and preprocessors to graph filters"
            )
        return self

    def __lshift__(self, ranker):
        if not isinstance(ranker, NodeRanking):
            raise Exception("pygrank can only shift rankers into filters")
        self.personalization_transform = ranker
        return ranker
