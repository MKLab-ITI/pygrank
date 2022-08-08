from pygrank.core import to_signal, NodeRanking, GraphSignal, GraphSignalGraph, GraphSignalData, assert_binary
from pygrank.algorithms.postprocess.postprocess import Postprocessor, Tautology


class Subgraph(Postprocessor):
    """Extracts induced subgraphs for non-zero node scores and places those scores in new signals on it."""

    def __init__(self, ranker: NodeRanking = None):
        """Initializes the postprocessor with a base ranker.

        Args:
            ranker: Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified.

        Example:
            >>> import pygrank as pg
            >>> graph, personalization, algorithm = ...
            >>> algorithm = pg.Subgraph(pg.Top(algorithm, 10))
            >>> top_10_subgraph = algorithm(graph, personalization).graph

        Example (same result):
            >>> import pygrank as pg
            >>> graph, personalization, algorithm = ...
            >>> algorithm = algorithm >> pg.Top(10) >> pg.Subgraph()
            >>> top_10_subgraph = algorithm(graph, personalization).graph
        """
        super().__init__(Tautology() if ranker is None else ranker)

    def transform(self, ranks: GraphSignal, *args, **kwargs):
        nodes = [u for u in ranks if ranks[u] != 0]
        graph = ranks.graph.subgraph(nodes)
        graph._pygrank_original_graph = ranks.graph
        return to_signal(graph, {u: ranks[u] for u in nodes})

    def rank(self, *args, **kwargs):
        ranks = self.ranker.rank(*args, **kwargs)
        return self.transform(ranks)

    def _reference(self):
        return "induced subgraph extraction for non-zero node scores"


class Supergraph(Postprocessor):
    """Reverts to full graphs from which `Subgraph` departed."""

    def __init__(self, ranker: NodeRanking = None):
        """Initializes the postprocessor with a base ranker.

        Args:
            ranker: Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified.

        Example:
            >>> import pygrank as pg
            >>> graph, personalization, algorithm, test = ...
            >>> algorithm = algorithm >> pg.Top(10) >> pg.Threshold() >> pg.Subgraph() >> pg.PageRank() >> pg.Supergraph()
            >>> top_10_reranked = algorithm(graph, personalization)  # top 10 non-zeroes ranked in their induced subgraph
            >>> print(pg.AUC(pg.to_signal(graph, test))(top_10_reranked))  # supergraph has returned to the original graph
        """
        super().__init__(Tautology() if ranker is None else ranker)

    def transform(self, ranks: GraphSignal, *args, **kwargs):
        return to_signal(ranks.graph._pygrank_original_graph, {u: ranks[u] for u in ranks})

    def rank(self, *args, **kwargs):
        ranks = self.ranker.rank(*args, **kwargs)
        return self.transform(ranks)

    def _reference(self):
        return "returned to subgraph's containing graph"
