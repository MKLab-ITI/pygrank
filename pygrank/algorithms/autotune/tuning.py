from pygrank.core.signals import GraphSignal, to_signal, NodeRanking, GraphSignalGraph, GraphSignalData
from typing import Tuple


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
