import numpy as np
from pygrank.measures.utils import Measure
from pygrank.core.signals import to_signal, GraphSignal
from pygrank.core import backend, GraphSignalGraph, GraphSignalData, BackendPrimitive
from pygrank.core.utils import preprocessor as default_preprocessor


class Unsupervised(Measure):
    def __init__(self, graph: GraphSignalGraph = None, preprocessor=None, **kwargs):
        """ Initializes the unsupervised measure.
           Args:
            graph: Optional. The graph on which to calculate the measure. If None (default) it is automatically
             extracted from graph signals passed for evaluation.
            preprocessor: Optional. Method to extract a scipy sparse matrix from a networkx graph.
                If None (default), pygrank.algorithms.utils.preprocessor is used with keyword arguments
                automatically extracted from the ones passed to this constructor, setting no normalization.

           Example:
               >>> import pygrank as pg
               >>> graph, seed_nodes, algorithm = ...
               >>> algorithm = pg.Normalize(algorithm)
               >>> scores = algorithm.rank(graph, seed_nodes)
               >>> conductance = pg.Conductance().evaluate(scores)
           """
        self.graph = graph
        if preprocessor is None and "normalization" not in kwargs:
            kwargs["normalization"] = "none"
        self.preprocessor = default_preprocessor(**kwargs) if preprocessor is None else preprocessor

    def to_numpy(self, scores: GraphSignalData = None):
        scores = to_signal(self.graph, scores)
        graph = scores.graph  # if None original graph this will end up obtaining the signal's graph
        return self.preprocessor(graph), scores.np

    def get_graph(self, scores: GraphSignalData = None):
        if scores is not None and isinstance(scores, GraphSignal):
            scores = to_signal(self.graph, scores)
            return scores.graph
        return self.graph

    def best_direction(self) -> int:
        ret = getattr(self.__class__, "__best_direction", None)
        if ret is None:
            import networkx as nx
            graph = nx.Graph([("A","B"), ("B","C"), ("C","A"), ("C", "D"), ("D","E"), ("E","F"), ("F","D")])
            ret = 1 if self.__class__(graph)(["A", "B", "C"]) \
                       > self.__class__(graph)(["A", "C", "F"]) else -1
            setattr(self.__class__, "__best_direction", ret)
        return ret


class Conductance(Unsupervised):
    """ Graph conductance (information flow) of scores.

    Assumes a fuzzy set of subgraphs whose nodes are included with probability proportional to their scores,
    as per the formulation of [krasanakis2019linkauc] and calculates E[outgoing edges] / E[internal edges] of
    the fuzzy rank subgraph. To avoid potential optimization towards filling the whole graph, the measure is
    evaluated to infinity if either denomator *or* the nominator is zero (this means that whole connected components
    should not be extracted).
    If scores assume binary values, E[.] becomes set size and this calculates the induced subgraph Conductance.
    """

    def __init__(self, graph: GraphSignalGraph = None, max_rank: float = 1, autofix=False, **kwargs):
        """ Initializes the Conductance measure.

        Args:
            max_rank: Optional. The maximum value scores can assume. To maintain a probabilistic formulation of
             conductance, this can be greater but not less than the maximum rank during evaluation. Default is 1.
             Pass algorithms through a normalization to ensure that this limit is not violated.
            autofix: Optional. If True, automatically normalizes scores by multiplying with max_rank / their maximum.
             If False (default) and the maximum score is greater than max_rank, an exception is thrown.

        Example:
            >>> import pygrank as pg
            >>> graph, seed_nodes, algorithm = ...
            >>> algorithm = pg.Normalize(algorithm)
            >>> scores = algorithm.rank(graph, seed_nodes)
            >>> conductance = pg.Conductance().evaluate(scores)

        Example (same conductance):
            >>> import pygrank as pg
            >>> graph, seed_nodes, algorithm = ...
            >>> scores = algorithm.rank(graph, seed_nodes)
            >>> conductance = pg.Conductance(autofix=True).evaluate(scores)
        """
        self.max_rank = max_rank
        self.autofix = autofix
        super().__init__(graph, **kwargs)

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        graph = self.get_graph(scores)
        if len(graph) == 0:
            return float('inf')
        adjacency, scores = self.to_numpy(scores)
        if backend.max(scores) > self.max_rank:
            if self.autofix:
                scores = scores * (self.max_rank / backend.max(scores))
            else:
                raise Exception("Normalize scores to be <= " + str(self.max_rank) + " for non-negative conductance")
        neighbors = backend.conv(scores, adjacency)
        internal_edges = backend.dot(neighbors, scores)
        external_edges = backend.dot(neighbors, self.max_rank-scores)
        if not graph.is_directed():
            external_edges += backend.dot(scores, backend.conv(self.max_rank-scores, adjacency))
            internal_edges *= 2
        if external_edges == 0:
            return float('inf')
        return backend.safe_div(external_edges, internal_edges, default=float('inf'))


class Density(Unsupervised):
    """ Extension of graph density that accounts for node scores.

    Assumes a fuzzy set of subgraphs whose nodes are included with probability proportional to their scores,
    as per the formulation of [krasanakis2019linkauc] and calculates E[internal edges] / E[possible edges] of
    the fuzzy rank subgraph.
    If scores assume binary values, E[.] becomes set size and this calculates the induced subgraph Density.
    """

    def __init__(self, graph: GraphSignalGraph = None, **kwargs):
        """ Initializes the Density measure.

        Args:
            graph: Optional. The graph on which to calculate the measure. If None (default) it is automatically
             extracted from graph signals passed for evaluation.

        Example:
            >>> import pygrank as pg
            >>> graph, seed_nodes, algorithm = ...
            >>> scores = algorithm.rank(graph, seed_nodes)
            >>> conductance = pg.Density().evaluate(scores)
        """
        super().__init__(graph, **kwargs)

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        if len(self.get_graph(scores)) == 0:
            return 0
        adjacency, scores = self.to_numpy(scores)
        neighbors = backend.conv(scores, adjacency)
        internal_edges = backend.dot(neighbors, scores)
        expected_edges = backend.sum(scores) ** 2 - backend.sum(scores ** 2) # without self-loops
        return backend.safe_div(internal_edges, expected_edges)


class Modularity(Unsupervised):
    """
    Extension of modularity that accounts for node scores.
    """
    
    def __init__(self,
                 graph: GraphSignalGraph = None,
                 max_rank: float = 1,
                 max_positive_samples: int = 2000,
                 seed: int = 0,
                 progress = lambda x: x):
        """ Initializes the Modularity measure with a sampling strategy that speeds up normal computations.

        Args:
            graph: Optional. The graph on which to calculate the measure. If None (default) it is automatically
             extracted from graph signals passed for evaluation.
            max_rank: Optional. Default is 1.
            max_positive_samples: Optional. The number of nodes with which to compute modularity. These are
             sampled uniformly from all graph nodes. If this is greater than the number of graph nodes,
             all nodes are used and the measure is deterministic. However,
             calculation time is O(max_positive_samples<sup>2</sup>) and thus a trade-off needs to be determined of time
             vs approximation quality. Effectively, the value should be high enough for max_positive_samples<sup>2</sup>
             to be comparable to the number of graph edges. Default is 2000.
            seed: Optional. Makes the evaluation seeded, for example to use in tuning. Default is 0.

        Example:
            >>> import pygrank as pg
            >>> graph, seed_nodes, algorithm = ...
            >>> scores = algorithm.rank(graph, seed_nodes)
            >>> modularity = pg.Modularity(max_positive_samples=int(graph.number_of_edges()**0.5)).evaluate(scores)
        """
        self.graph = graph
        self.max_positive_samples = max_positive_samples
        self.max_rank = max_rank
        self.seed = seed
        self.progress = progress

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        scores = to_signal(self.graph, scores)
        graph = scores.graph
        positive_candidates = list(graph)
        if len(positive_candidates) > self.max_positive_samples:
            np.random.seed(self.seed)
            positive_candidates = np.random.choice(positive_candidates, self.max_positive_samples)
        m = graph.number_of_edges()
        if m == 0:
            return 0
        Q = 0
        for v in self.progress(positive_candidates):
            for u in positive_candidates:
                Avu = 1 if graph.has_edge(v, u) else 0
                Avu -= graph.degree[v]*graph.degree[u]/2/m
                Q += Avu*(scores[v]/self.max_rank)*(scores[u]/self.max_rank)
        return Q/2/m
