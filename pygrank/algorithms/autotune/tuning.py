from pygrank.core.signals import GraphSignal, to_signal, NodeRanking, GraphSignalGraph, GraphSignalData, no_signal
from pygrank.core.utils import preprocessor, ensure_used_args, remove_used_args
from pygrank.algorithms.autotune.optimization import optimize, evolutionary_optimize, incremental_optimizer
from pygrank.measures import Measure, AUC
from pygrank.measures.utils import split
from typing import Callable, Iterable, Tuple, Optional
from pygrank.core import backend


class Tuner(NodeRanking):
    def tune(self,
             graph: GraphSignalGraph = None,
             personalization: GraphSignalData = None,
             *args, **kwargs) -> NodeRanking:
        return self._tune(graph, personalization, *args, **kwargs)[0]

    def rank(self,
             graph: GraphSignalGraph = None,
             personalization: GraphSignalData = None,
             *args, **kwargs) -> GraphSignal:
        ranker, personalization = self._tune(graph, personalization, *args, **kwargs)
        return ranker.rank(graph, personalization, *args, **kwargs)

    def _tune(self,
              graph: GraphSignalGraph = None,
              personalization: GraphSignalData = None,
              *args, **kwargs) -> Tuple[NodeRanking, GraphSignal]:
        raise Exception("Tuners should implement a _tune method")

