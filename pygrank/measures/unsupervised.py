import warnings
import networkx as nx
import numpy as np
from pygrank.measures.utils import Measure
from pygrank.algorithms.utils import to_signal


class Conductance(Measure):
    """ Graph conductance (information flow) of ranks.

    Assumes a fuzzy set of subgraphs whose nodes are included with probability proportional to their ranks,
    as per the formulation of [krasanakis2019linkauc] and calculates E[outgoing edges] / E[internal edges] of
    the fuzzy rank subgraph.
    If ranks assume binary values, E[.] becomes set size and this calculates the induced subgraph Conductance.
    """

    def __init__(self, graph=None, max_rank=1):
        """ Initializes the Conductance metric.

        Args:
            graph: Optional. The graph on which to calculate the metric. If None (default) it is automatically extracted
             from graph signals passed for evaluation.
            max_rank: Optional. The maximum value ranks can assume. To maintain a probabilistic formulation of
             conductance, this can be greater but not less than the maximum rank during evaluation. Default is 1.

        Example:
            >>> from pygrank.metrics.unsupervised import Conductance
            >>> from pygrank.algorithms.postprocess import Normalize
            >>> graph, seed_nodes, algorithm = ...
            >>> algorithm = Normalize(algorithm)
            >>> ranks = algorithm.rank(graph, seed_nodes)
            >>> conductance = Conductance().evaluate(ranks)
        """
        self.graph = graph
        self.max_rank = max_rank

    def evaluate(self, ranks):
        ranks = to_signal(self.graph, ranks)
        graph = ranks.graph
        if max(ranks.values()) > self.max_rank:
            warnings.warn("Normalize ranks to be <= " + str(self.max_rank)
                          + " to guarantee correct probabilistic formulation", stacklevel=2)
        external_edges = sum(ranks.get(i, 0)*(self.max_rank-ranks.get(j, 0)) for i, j in graph.edges())
        internal_edges = sum(ranks.get(i, 0)*ranks.get(j, 0) for i, j in graph.edges())
        if internal_edges > graph.number_of_edges()/2:
            internal_edges = graph.number_of_edges()-internal_edges # user the smallest partition as reference
        if not graph.is_directed():
            external_edges += sum(ranks.get(j, 0) * (self.max_rank - ranks.get(i, 0)) for i, j in graph.edges())
            internal_edges *= 2
        return external_edges / internal_edges if internal_edges !=0 else float('inf')


class Density(Measure):
    """ Extension of graph density that can account for ranks.

    Assumes a fuzzy set of subgraphs whose nodes are included with probability proportional to their ranks,
    as per the formulation of [krasanakis2019linkauc] and calculates E[internal edges] / E[possible edges] of
    the fuzzy rank subgraph.
    If ranks assume binary values, E[.] becomes set size and this calculates the induced subgraph Density.
    """

    def __init__(self, graph=None):
        """ Initializes the Density metric.

        Args:
            graph: Optional. The graph on which to calculate the metric. If None (default) it is automatically extracted
             from graph signals passed for evaluation.

        Example:
            >>> from pygrank.metrics.unsupervised import Density
            >>> graph, seed_nodes, algorithm = ...
            >>> ranks = algorithm.rank(graph, seed_nodes)
            >>> conductance = Density().evaluate(ranks)
        """
        self.graph = graph

    def evaluate(self, ranks):
        ranks = to_signal(self.graph, ranks)
        graph = ranks.graph
        internal_edges = sum(ranks.get(i, 0) * ranks.get(j, 0) for i,j  in graph.edges())
        expected_edges = sum(ranks.values()) ** 2 - sum(rank ** 2 for rank in ranks.values()) # without self-loops
        if internal_edges == 0:
            return 0
        return internal_edges / expected_edges


class Modularity(Measure):
    def __init__(self, graph, max_rank=1, max_positive_samples=2000):
        self.graph = graph
        self.max_positive_samples = max_positive_samples
        self.max_rank = max_rank

    def evaluate(self, ranks):
        ranks = to_signal(self.graph, ranks)
        graph = ranks.graph
        positive_candidates = list(graph)
        if len(positive_candidates) > self.max_positive_samples:
            positive_candidates = np.random.choice(positive_candidates, self.max_positive_samples)
        m = graph.number_of_edges()
        Q = 0
        for v in positive_candidates:
            for u in positive_candidates:
                Avu = 1 if graph.has_edge(v,u) else 0
                Avu -= graph.degree[v]*graph.degree[u]/2/m
                Q += Avu*(ranks[v]/self.max_rank)*(ranks[u]/self.max_rank)
        return Q/2/m
