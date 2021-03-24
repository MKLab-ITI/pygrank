import warnings
import networkx as nx
import numpy as np
from .utils import (__Metric__)


class Mean(__Metric__):
    def __init__(self):
        pass

    def evaluate(self, ranks):
        return sum(ranks.values())/len(ranks)


class Conductance(__Metric__):
    """ Graph conductance (information flow) of ranks.

    Assumes a fuzzy set of subgraphs whose nodes are included with probability proportional to their ranks,
    as per the formulation of [krasanakis2019linkauc] and calculates E[outgoing edges] / E[internal edges] of
    the fuzzy rank subgraph.
    If ranks assume binary values, E[.] becomes set size and this calculates the induced subgraph Conductance.
    """

    def __init__(self, G, max_rank=1):
        """ Initializes the Conductance metric.

        Attributes:
            G: The graph on which to calculate the metric.
            max_rank: Optional. The maximum value ranks can assume. To maintain a probabilistic formulation of
             conductance, this can be greater but not less than the maximum rank during evaluation. Default is 1.

        Example:
            >>> from pygrank.metrics.unsupervised import Conductance
            >>> from pygrank.algorithms.postprocess import Normalize
            >>> G, seed_nodes, algorithm = ...
            >>> algorithm = Normalize(algorithm)
            >>> ranks = algorithm.rank(G, seed_nodes)
            >>> conductance = Conductance(G).evaluate(ranks)
        """
        self.G = G
        self.max_rank = max_rank

    def evaluate(self, ranks):
        if max(ranks.values()) > self.max_rank:
            warnings.warn("Normalize ranks to be <= " + str(self.max_rank)
                          + " to guarantee correct probabilistic formulation", stacklevel=2)
        external_edges = sum(ranks.get(i, 0)*(self.max_rank-ranks.get(j, 0)) for i, j in self.G.edges())
        internal_edges = sum(ranks.get(i, 0)*ranks.get(j, 0) for i, j in self.G.edges())
        if internal_edges > self.G.number_of_edges()/2:
            internal_edges = self.G.number_of_edges()-internal_edges # user the smallest partition as reference
        if not self.G.is_directed():
            external_edges += sum(ranks.get(j, 0) * (self.max_rank - ranks.get(i, 0)) for i, j in self.G.edges())
            internal_edges *= 2
        if internal_edges == 0:
            return float('inf')
        return external_edges / internal_edges


class Density(__Metric__):
    """ Extension of graph density that can account for ranks.

    Assumes a fuzzy set of subgraphs whose nodes are included with probability proportional to their ranks,
    as per the formulation of [krasanakis2019linkauc] and calculates E[internal edges] / E[possible edges] of
    the fuzzy rank subgraph.
    If ranks assume binary values, E[.] becomes set size and this calculates the induced subgraph Density.
    """

    def __init__(self, G):
        """ Initializes the Density metric.

        Attributes:
            G: The graph on which to calculate the metric.

        Example:
            >>> from pygrank.metrics.unsupervised import Density
            >>> G, seed_nodes, algorithm = ...
            >>> ranks = algorithm.rank(G, seed_nodes)
            >>> conductance = Density(G).evaluate(ranks)
        """
        self.G = G

    def evaluate(self, ranks):
        internal_edges = sum(ranks.get(i, 0) * ranks.get(j, 0) for i,j  in self.G.edges())
        expected_edges = sum(ranks.values()) ** 2 - sum(rank ** 2 for rank in ranks.values()) # without self-loops
        if internal_edges == 0:
            return 0
        return internal_edges / expected_edges


class Modularity(__Metric__):
    def __init__(self, G, max_rank=1, max_positive_samples=2000):
        self.G = G
        self.max_positive_samples = max_positive_samples
        self.max_rank = max_rank

    def evaluate(self, ranks):
        positive_candidates = list(self.G)
        if len(positive_candidates) > self.max_positive_samples:
            positive_candidates = np.random.choice(positive_candidates, self.max_positive_samples)
        m = self.G.number_of_edges()
        Q = 0
        for v in positive_candidates:
            for u in positive_candidates:
                Avu = 1 if self.G.has_edge(v,u) else 0
                Avu -= self.G.degree[v]*self.G.degree[u]/2/m
                Q += Avu*(ranks[v]/self.max_rank)*(ranks[u]/self.max_rank)
        return Q/2/m
