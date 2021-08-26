from pygrank.core.signals import GraphSignal, to_signal, NodeRanking
from pygrank.algorithms.utils import ensure_used_args, remove_used_args
from pygrank.algorithms.autotune.optimization import optimize
from pygrank.measures import Supervised, AUC
import random
from typing import Callable
from pygrank.core import backend
from pygrank.algorithms.autotune.tuning import Tuner


class HopTuner(Tuner):
    """
    Tunes a GenericGraphFilter specific measure by splitting the personalization
    in training and test sets and measuring the similarity of hops at given number of steps
    away.
    """
    def __init__(self, ranker_generator: Callable[[list], NodeRanking] = None,
                 measure: Callable[[GraphSignal], Supervised] = AUC,
                 tuning_backend: str = None,
                 **kwargs):
        """
        Instantiates the tuning mechanism.
        Args:
            ranker_generator: A callable that constructs a ranker based on a list of parameters.
                If None (default) then a pygrank.algorithms.learnable.GenericGraphFilter
                is constructed with automatic normalization and assuming immutability (this is the most common setting).
                These parameters can be overriden and other ones can be passed to the algorithm'personalization constructor simply
                by including them in kwargs.
            measure: Callable to constuct a supervised measure with given known node scores.
            tuning_backend: Specifically switches to a designted backend for the tuning process before restoring
                the previous one to perform the actual ranking. If None (default), this functionality is ignored.
            kwargs: Additional arguments are passed to the automatically instantiated GenericGraphFilter.

        Example:
            >>> import pygrank as pg
            >>> graph, personalization = ...
            >>> tuner = pg.HopTuner(measure=AUC)
            >>> ranks = tuner.rank(graph, personalization)
        """
        if ranker_generator is None:
            from pygrank.algorithms import GenericGraphFilter
            ranker_generator = lambda params: GenericGraphFilter(params, **kwargs)
        else:
            ensure_used_args(kwargs, [])
        self.ranker_generator = ranker_generator
        self.measure = measure
        self.tuning_backend = tuning_backend

    def _tune(self, graph=None, personalization=None, *args, **kwargs):
        previous_backend = backend.backend_name()
        personalization = to_signal(graph, personalization)
        graph = personalization.graph
        if self.tuning_backend is not None and self.tuning_backend != previous_backend:
            backend.load_backend(self.tuning_backend)
        backend_personalization = to_signal(graph, backend.to_array(personalization.np))
        measure = self.measure(backend_personalization)
        rand_test = backend.to_array([int(random.random()+0.5) for _ in range(1000)])
        measure_worst = self.measure(rand_test)(1-rand_test)
        measure_best = self.measure(rand_test)(rand_test)

        best_parameters = [1]*100
        M = self.ranker_generator(best_parameters).preprocessor(graph)
        propagated = backend_personalization.np
        for i in range(len(best_parameters)):
            best_parameters[i] = ((measure(propagated)-(measure_best+measure_worst)/2))*2/(measure_best-measure_worst)
            propagated = backend.conv(propagated, M)
        for i in range(1, len(best_parameters)):
            assert best_parameters[i] >= 0
            best_parameters[i] = best_parameters[i-1] * (best_parameters[i]+1)/2
        #print(best_parameters)

        if self.tuning_backend is not None and self.tuning_backend != previous_backend:
            backend.load_backend(previous_backend)
        return self.ranker_generator(best_parameters), personalization
