from pygrank.measures.measure import Measure
from pygrank.core.signals import to_signal, GraphSignal
from pygrank.core import backend, GraphSignalGraph, GraphSignalData, BackendPrimitive
from pygrank.core.utils import preprocessor as default_preprocessor


class Unsupervised(Measure):
    def __init__(self, graph: GraphSignalGraph = None, preprocessor=None, **kwargs):
        """Initializes the unsupervised measure.
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
        self.preprocessor = (
            default_preprocessor(**kwargs) if preprocessor is None else preprocessor
        )

    def to_numpy(self, scores: GraphSignalData = None):
        scores = to_signal(self.graph, scores)
        graph = (
            scores.graph
        )  # if None original graph this will end up obtaining the signal's graph
        adj = self.preprocessor(graph)
        # if adj.__class__.__name__ == "Adjacency":
        #    adj = adj._array
        return adj, scores.np

    def get_graph(self, scores: GraphSignalData = None):
        if scores is not None and isinstance(scores, GraphSignal):
            scores = to_signal(self.graph, scores)
            return scores.graph
        return self.graph

    def best_direction(self) -> int:
        ret = getattr(self.__class__, "__best_direction", None)
        if ret is None:
            import networkx as nx

            graph = nx.Graph(
                [
                    ("A", "B"),
                    ("B", "C"),
                    ("C", "A"),
                    ("C", "D"),
                    ("D", "E"),
                    ("E", "F"),
                    ("F", "D"),
                ]
            )
            ret = (
                1
                if self.__class__(graph)(["A", "B", "C"])
                > self.__class__(graph)(["A", "C", "F"])
                else -1
            )
            setattr(self.__class__, "__best_direction", ret)
        return ret
