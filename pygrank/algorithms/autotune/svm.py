from pygrank.core.signals import GraphSignal, to_signal, NodeRanking, no_signal, GraphSignalGraph, GraphSignalData
from pygrank.measures import Measure, AUC
from pygrank.measures.utils import split
from typing import Callable, Iterable
from pygrank.core import backend
from pygrank.algorithms.autotune.tuning import Tuner


class SVMFilter(NodeRanking):
    def __init__(self, rankers, weights):
        self.rankers = rankers
        self.weights = weights

    def rank(self,
             graph: GraphSignalGraph = None,
             personalization: GraphSignalData = None,
             *args, **kwargs) -> GraphSignal:
        result = 0
        for ranker, weight in zip(self.rankers, self.weights):
            result = ranker(graph, personalization, *args, **kwargs) * weight + result
        return result

"""
class SVM(Tuner):
    def __init__(self,
                 rankers,
                 measure: Callable[[GraphSignal, GraphSignal], Measure] = AUC,
                 fraction_of_training: float = 0.5,
                 combined_prediction: bool = True,
                 tuning_backend: str = None):
        self.measure = measure
        self.fraction_of_training = fraction_of_training
        self.combined_prediction = combined_prediction
        self.tuning_backend = tuning_backend

    def _tune(self, graph=None, personalization=None, *args, **kwargs):
        previous_backend = backend.backend_name()
        personalization = to_signal(graph, personalization)
        if self.tuning_backend is not None and self.tuning_backend != previous_backend:
            backend.load_backend(self.tuning_backend)
        backend_personalization = to_signal(graph, backend.to_array(personalization.np))
        training, validation = split(backend_personalization, self.fraction_of_training)
        measure = self.measure(validation, training)
        best_value = -float('inf')
        best_ranker = None
        for ranker in self.rankers:
            value = measure.best_direction() * measure.evaluate(ranker.rank(training, *args, **kwargs))
            if value > best_value:
                best_value = value
                best_ranker = ranker
        if self.tuning_backend is not None and self.tuning_backend != previous_backend:
            backend.load_backend(previous_backend)
            # TODO: make training back-propagate through tensorflow for combined_prediction==False
        return best_ranker, personalization if self.combined_prediction else training

    def
"""