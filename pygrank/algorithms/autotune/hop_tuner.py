import pygrank.algorithms.postprocess.postprocess
from pygrank.core.signals import GraphSignal, to_signal, NodeRanking
from pygrank.algorithms.utils import ensure_used_args, arnoldi_iteration
from pygrank.measures import Supervised, AUC
from typing import Callable
from pygrank.core import backend
from pygrank.algorithms.autotune.tuning import Tuner
from pygrank.algorithms.autotune.optimization import optimize
from pygrank.measures.utils import split
from pygrank.algorithms.postprocess.postprocess import Tautology
import numpy as np


class HopTuner(Tuner):
    """
    Tunes a GenericGraphFilter specific measure by splitting the personalization
    in training and test sets and measuring the similarity of hops at given number of steps
    away.

    """
    def __init__(self, ranker_generator: Callable[[list], NodeRanking] = None,
                 measure: Callable[[GraphSignal, GraphSignal], Supervised] = AUC,
                 basis: str = "krylov",
                 tuning_backend: str = None,
                 autoregression: int = 0,
                 num_parameters: int = 10,
                 tunable_offset: Callable[[GraphSignal, GraphSignal], Supervised] = AUC,
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
            tunable_offset: If None, no offset is added to estimated parameters. Otherwise, a supervised measure
                generator (e.g. a supervised measure class) can be passed. Default is `pygrank.AUC`.
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
        self.tunable_offset = tunable_offset
        self.basis = basis

    def _tune(self, graph=None, personalization=None, *args, **kwargs):
        previous_backend = backend.backend_name()
        personalization = to_signal(graph, personalization)
        graph = personalization.graph
        if self.tuning_backend is not None and self.tuning_backend != previous_backend:
            backend.load_backend(self.tuning_backend)
        backend_personalization = to_signal(personalization, backend.to_array(personalization.np))
        #training, validation = split(backend_personalization, 0.8)
        #training2, validation2 = split(backend_personalization, 0.6)
        #measure_weights = [1, 1, 1, 1, 1]
        #propagated = [training.np, validation.np, backend_personalization.np, training2.np, validation2.np]

        measure_values = [None] * (self.num_parameters+self.autoregression)
        M = self.ranker_generator(measure_values).preprocessor(graph)

        #for _ in range(10):
        #    backend_personalization.np = backend.conv(backend_personalization.np, M)
        training, validation = split(backend_personalization, 0.5)
        propagated = [training.np, validation.np]
        #measures = [self.measure(backend_personalization, training), self.measure(backend_personalization, validation)]
        measures = [self.measure(backend_personalization)]*len(propagated)

        if self.basis == "krylov":
            for i in range(len(measure_values)):
                measure_values[i] = [measure(p) for p, measure in zip(propagated, measures)]
                propagated = [backend.conv(p, M) for p in propagated]
        elif self.basis == "arnoldi":
            basis = [arnoldi_iteration(M, p, len(measure_values))[0] for p in propagated]
            for i in range(len(measure_values)):
                measure_values[i] = [measure(base[:,i]) for base, measure in zip(basis, measures)]
        else:
            raise Exception("Invalid basis in hop tuner. Should be either 'krylov' or 'arnoldi'")

        measure_values = backend.to_array(measure_values)
        mean_value = np.mean(measure_values, axis=0)
        measure_values = measure_values-mean_value
        best_parameters = measure_values
        measure_weights = [1] * measure_values.shape[1]
        if self.autoregression != 0:
            #vals2 = -measure_values-mean_value
            #measure_values = np.concatenate([measure_values, vals2-np.mean(vals2, axis=0)], axis=1)
            window = backend.repeat(1./self.autoregression, self.autoregression)
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
                for i in range(len(measure_values) - len(window)-2, -1, -1):
                    parameters[i, :] = np.dot((window), measure_values[(i+1):(i+len(window)+1), :])
                errors = (parameters - measure_values) * measure_weights / np.sum(measure_weights)
                for j in range(len(window)):
                    gradient = 0
                    for i in range(len(measure_values) - len(window)-1):
                        gradient += np.dot(measure_values[i+j+1, :], errors[i, :])
                    momentum[j] = beta1*momentum[j] + (1-beta1)*gradient#*np.sign(window[j])
                    rms[j] = beta2*rms[j] + (1-beta2)*gradient*gradient
                    window[j] -= 0.01*momentum[j] / (1-beta1t) / ((rms[j]/(1-beta2t))**0.5 + 1.E-8)
                    #window[j] -= 0.01*gradient*np.sign(window[j])
                error = np.mean(np.abs(errors))
                if abs(error-prev_error)/error < 1.E-6:
                    best_parameters = parameters
                    break
        best_parameters = np.mean(best_parameters[:self.num_parameters, :] * measure_weights, axis=1) + np.mean(mean_value)
        #best_parameters /= np.max(best_parameters)

        if self.tunable_offset is not None:
            div = np.max(best_parameters)
            if div != 0:
                best_parameters /= div
            measure = self.tunable_offset(validation, training)
            best_offset = optimize(
                lambda params: - measure(self._run(training, [best_parameters[i]*params[0]**i for i in range(len(best_parameters))], *args, **kwargs)),
                #lambda params: - measure.evaluate(self._run(training, best_parameters + params[0], *args, **kwargs)),
                max_vals=[1], min_vals=[0], deviation_tol=0.005, parameter_tol=1, partitions=5, divide_range=2)#"shrinking")
            #best_parameters += best_offset[0]
            best_parameters = [best_parameters[i]*best_offset[0]**i for i in range(len(best_parameters))]

        if np.sum(np.abs(best_parameters)) != 0:
            best_parameters /= np.sum(np.abs(best_parameters))/len(best_parameters)
        if self.tuning_backend is not None and self.tuning_backend != previous_backend:
            backend.load_backend(previous_backend)
        if self.basis == "arnoldi":
            return Tautology(), self._run(personalization, best_parameters, *args, **kwargs)
        return self.ranker_generator(best_parameters), personalization

    def _run(self, personalization: GraphSignal, params: list, *args, **kwargs):
        div = np.sum(np.abs(params)) / backend.length(params)
        if div != 0:
            params = params / div
        if self.basis == "arnoldi":
            M = self.ranker_generator(params).preprocessor(personalization.graph)
            base = arnoldi_iteration(M, personalization.np, len(params))[0]
            ret = 0
            for i in range(len(params)):
                ret = ret + params[i]*base[:,i]
            return to_signal(personalization, ret)
        return self.ranker_generator(params).rank(personalization, *args, **kwargs)
