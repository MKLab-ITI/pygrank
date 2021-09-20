import warnings
import numpy as np
from pygrank.measures.utils import Measure
from pygrank.core.signals import to_signal
from pygrank.core import backend, GraphSignalGraph, GraphSignalData, BackendPrimitive


class Unsupervised(Measure):
    pass


class Conductance(Unsupervised):
    """ Graph conductance (information flow) of scores.

    Assumes a fuzzy set of subgraphs whose nodes are included with probability proportional to their scores,
    as per the formulation of [krasanakis2019linkauc] and calculates E[outgoing edges] / E[internal edges] of
    the fuzzy rank subgraph.
    If scores assume binary values, E[.] becomes set size and this calculates the induced subgraph Conductance.
    """

    def __init__(self, graph: GraphSignalGraph = None, max_rank: float = 1):
        """ Initializes the Conductance measure.

        Args:
            graph: Optional. The graph on which to calculate the measure. If None (default) it is automatically extracted
             from graph signals passed for evaluation.
            max_rank: Optional. The maximum value scores can assume. To maintain a probabilistic formulation of
             conductance, this can be greater but not less than the maximum rank during evaluation. Default is 1.

        Example:
            >>> import pygrank as pg
            >>> graph, seed_nodes, algorithm = ...
            >>> algorithm = pg.Normalize(algorithm)
            >>> scores = algorithm.rank(graph, seed_nodes)
            >>> conductance = pg.Conductance().evaluate(scores)
        """
        self.graph = graph
        self.max_rank = max_rank

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        scores = to_signal(self.graph, scores)
        graph = scores.graph
        if max(scores.values()) > self.max_rank:
            warnings.warn("Normalize scores to be <= " + str(self.max_rank)
                          + " to guarantee correct probabilistic formulation", stacklevel=2)
        external_edges = sum(scores.get(i, 0)*(self.max_rank-scores.get(j, 0)) for i, j in graph.edges())
        internal_edges = sum(scores.get(i, 0)*scores.get(j, 0) for i, j in graph.edges())
        if internal_edges > graph.number_of_edges()/2:
            internal_edges = graph.number_of_edges()-internal_edges # user the smallest partition as reference
        if not graph.is_directed():
            external_edges += sum(scores.get(j, 0) * (self.max_rank - scores.get(i, 0)) for i, j in graph.edges())
            internal_edges *= 2
        return external_edges / internal_edges if internal_edges != 0 else float('inf')


class Density(Unsupervised):
    """ Extension of graph density that accounts for node scores.

    Assumes a fuzzy set of subgraphs whose nodes are included with probability proportional to their scores,
    as per the formulation of [krasanakis2019linkauc] and calculates E[internal edges] / E[possible edges] of
    the fuzzy rank subgraph.
    If scores assume binary values, E[.] becomes set size and this calculates the induced subgraph Density.
    """

    def __init__(self, graph: GraphSignalGraph = None):
        """ Initializes the Density measure.

        Args:
            graph: Optional. The graph on which to calculate the measure. If None (default) it is automatically extracted
             from graph signals passed for evaluation.

        Example:
            >>> import pygrank as pg
            >>> graph, seed_nodes, algorithm = ...
            >>> scores = algorithm.rank(graph, seed_nodes)
            >>> conductance = pg.Density().evaluate(scores)
        """
        self.graph = graph

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        scores = to_signal(self.graph, scores)
        graph = scores.graph
        internal_edges = sum(scores.get(i, 0) * scores.get(j, 0) for i,j in graph.edges())
        expected_edges = backend.sum(scores.np) ** 2 - backend.sum(scores.np ** 2) # without self-loops
        if internal_edges == 0:
            return 0
        return internal_edges / expected_edges


class Modularity(Unsupervised):
    """
    Extension of modularity that accounts for node scores.
    """
    
    def __init__(self,
                 graph: GraphSignalGraph = None,
                 max_rank: float = 1,
                 max_positive_samples: int = 2000,
                 seed: int = 0):
        self.graph = graph
        self.max_positive_samples = max_positive_samples
        self.max_rank = max_rank
        self.seed = seed

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
        for v in positive_candidates:
            for u in positive_candidates:
                Avu = 1 if graph.has_edge(v,u) else 0
                Avu -= graph.degree[v]*graph.degree[u]/2/m
                Q += Avu*(scores[v]/self.max_rank)*(scores[u]/self.max_rank)
        return Q/2/m
