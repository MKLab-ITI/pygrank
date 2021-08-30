from pygrank.core.signals import GraphSignal, to_signal, NodeRanking
from pygrank.algorithms.utils import ensure_used_args
from pygrank.measures import Supervised, AUC
import random
from typing import Callable
from pygrank.core import backend
from pygrank.algorithms.autotune.tuning import Tuner
from pygrank.measures.utils import split


class HopTuner(Tuner):
    """
    Tunes a GenericGraphFilter specific measure by splitting the personalization
    in training and test sets and measuring the similarity of hops at given number of steps
    away.
    """
    def __init__(self, ranker_generator: Callable[[list], NodeRanking] = None,
                 measure: Callable[[GraphSignal], Supervised] = AUC,
                 fraction_of_training: float = 0.8,
                 tuning_backend: str = None,
                 autoregression: int = 5,
                 num_parameters: int = 10,
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
            fraction_of_training: A number in (0,1) indicating how to split provided graph signals into training and
                validaton ones by randomly sampling training nodes to meet the required fraction of all graph nodes.
                Numbers outside this range can also be used (not recommended without specific reason) per the
                conventions of `pygrank.split(...)`. Default is 0.8.
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
        self.fraction_of_training = fraction_of_training
        self.tuning_backend = tuning_backend
        self.autoregression = autoregression
        self.num_parameters = num_parameters

    def _tune(self, graph=None, personalization=None, *args, **kwargs):
        previous_backend = backend.backend_name()
        personalization = to_signal(graph, personalization)
        graph = personalization.graph
        if self.tuning_backend is not None and self.tuning_backend != previous_backend:
            backend.load_backend(self.tuning_backend)
        backend_personalization = to_signal(graph, backend.to_array(personalization.np))
        rand_test = backend.to_array([int(random.random()+0.5) for _ in range(1000)])
        measure_worst = self.measure(rand_test)(1-rand_test)
        measure_best = self.measure(rand_test)(rand_test)
        training, validation = backend_personalization, backend_personalization#split(backend_personalization, self.fraction_of_training)
        measure = self.measure(validation)#, training if training != validation else None)

        measure_values = [1]*(self.num_parameters+self.autoregression)
        M = self.ranker_generator(measure_values).preprocessor(graph)
        propagated = training.np
        for i in range(len(measure_values)):
            measure_values[i] = ((measure(propagated)-(measure_best+measure_worst)/2))*2/(measure_best-measure_worst)#*0.5
            #measure_values[i] += ((measure2(propagated)-(measure_best+measure_worst)/2))*2/(measure_best-measure_worst)*0.5
            propagated = backend.conv(propagated, M)
        measure_values[0] = 1

        if self.autoregression == 0:
            best_parameters = [val for val in measure_values]
        else:
            mean_measure = sum(measure_values)/len(measure_values)
            measure_values = [val-mean_measure for val in measure_values]
            window = [1]*self.autoregression
            error = float('inf')
            while True:
                prev_error = error
                best_parameters = [val for val in measure_values]
                for i in range(len(measure_values) - len(window)-1, -1, -1):
                    best_parameters[i] = sum(best_parameters[i+j]*window[j] for j in range(len(window)))
                errors = [best_parameters[i]-measure_values[i] for i in range(len(best_parameters))]
                for j in range(len(window)):
                    for j in range(len(window)):
                        window[j] -= 0.01*sum(best_parameters[i+j]*errors[i] for i in range(len(measure_values) - len(window) - 1))
                window_sum = sum(window)
                window = [val/window_sum for val in window]
                error = sum(abs(val) for val in errors)/len(errors)
                if (prev_error-error) < 1.E-6:
                    break
            best_parameters = [val+mean_measure for val in best_parameters]
            best_parameters_sum = sum(best_parameters)
            best_parameters = [val/best_parameters_sum for val in best_parameters]

        if self.tuning_backend is not None and self.tuning_backend != previous_backend:
            backend.load_backend(previous_backend)
        return self.ranker_generator(best_parameters), personalization
