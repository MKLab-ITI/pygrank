from pygrank.core.signals import GraphSignal, to_signal, NodeRanking
from pygrank.algorithms.utils import ensure_used_args
from pygrank.measures import Supervised, AUC
from typing import Callable
from pygrank.core import backend
from pygrank.algorithms.autotune.tuning import Tuner
from pygrank.algorithms.autotune.optimization import optimize
from pygrank.measures.utils import split
import numpy as np


class HopTuner(Tuner):
    """
    Tunes a GenericGraphFilter specific measure by splitting the personalization
    in training and test sets and measuring the similarity of hops at given number of steps
    away.
    """
    def __init__(self, ranker_generator: Callable[[list], NodeRanking] = None,
                 measure: Callable[[GraphSignal], Supervised] = AUC,
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
        self.autoregression = autoregression
        self.num_parameters = num_parameters

    def _tune(self, graph=None, personalization=None, *args, **kwargs):
        previous_backend = backend.backend_name()
        personalization = to_signal(graph, personalization)
        graph = personalization.graph
        if self.tuning_backend is not None and self.tuning_backend != previous_backend:
            backend.load_backend(self.tuning_backend)
        backend_personalization = to_signal(graph, backend.to_array(personalization.np))
        #training, validation = split(backend_personalization, 0.8)
        #training2, validation2 = split(backend_personalization, 0.6)
        #measure_weights = [1, 1, 1, 1, 1]
        #propagated = [training.np, validation.np, backend_personalization.np, training2.np, validation2.np]

        training, validation = split(backend_personalization, 0.75)
        propagated = [training.np, validation.np, backend_personalization.np]
        measures = [self.measure(backend_personalization, None)]*3
        measure_weights = [1, 1, 1]

        measure_values = [None] * (self.num_parameters+self.autoregression)
        M = self.ranker_generator(measure_values).preprocessor(graph)
        for i in range(len(measure_values)):
            measure_values[i] = [measure(p) for p, measure in zip(propagated, measures)]
            #measure_values[i] += [val-0.5 for val in measure_values[i]]
            propagated = [backend.conv(p, M) for p in propagated]

        measure_values = np.array(measure_values)
        mean_value = np.mean(measure_values, axis=0)
        measure_values -= mean_value
        best_parameters = measure_values
        if self.autoregression != 0:
            window = np.repeat(1./self.autoregression, self.autoregression)
            beta1 = 0.9
            beta2 = 0.999
            beta1t = 1
            beta2t = 1
            rms = window*0
            momentum = window*0
            error = float('inf')
            while True:
                beta1t *= beta1
                beta2t *= beta2
                prev_error = error
                parameters = np.copy(measure_values)
                for i in range(len(measure_values) - len(window)-1, -1, -1):
                    parameters[i, :] = np.dot(np.abs(window), parameters[i:(i+len(window)), :])
                errors = (parameters - measure_values) * measure_weights / np.sum(measure_weights)
                for j in range(len(window)):
                    gradient = 0
                    for i in range(len(measure_values) - len(window)-1):
                        gradient += np.dot(parameters[i+j, :], errors[i, :])
                    momentum[j] = beta1*momentum[j] + (1-beta1)*gradient*np.sign(window[j])
                    rms[j] = beta2*rms[j] + (1-beta2)*gradient*gradient
                    window[j] -= 0.01*momentum[j] / (1-beta1t) / ((rms[j]/(1-beta2t))**0.5 + 1.E-8)
                    #window[j] -= 0.01*gradient*np.sign(window[j])
                error = np.mean(np.abs(errors))
                if abs(error-prev_error) < 1.E-6:
                    best_parameters = parameters
                    break
        best_parameters = (np.mean(best_parameters[:self.num_parameters, :] * measure_weights, axis=1) + np.mean(mean_value))
        best_parameters = best_parameters-np.min(best_parameters)
        """
        training, validation = split(backend_personalization, 0.8, seed=1)
        offset_fitness = lambda params: -self.measure(validation, training)(self.ranker_generator((best_parameters-params[0])*len(best_parameters)/np.sum(np.abs(best_parameters-params[0])))(training))
        offset = optimize(offset_fitness, max_vals=[0.5], min_vals=[-2], divide_range=1.5, deviation_tol=0.001)
        print("found offset", offset)
        best_parameters -= offset
        if np.sum(np.abs(best_parameters)) != 0:
            best_parameters /= np.sum(np.abs(best_parameters))/len(best_parameters)

        if self.tuning_backend is not None and self.tuning_backend != previous_backend:
            backend.load_backend(previous_backend)
        """
        return self.ranker_generator(best_parameters), personalization