from pygrank.core.signals import GraphSignal, to_signal, NodeRanking, GraphSignalGraph, GraphSignalData, no_signal
from pygrank.core.utils import preprocessor, ensure_used_args, remove_used_args
from pygrank.algorithms.autotune.optimization import optimize
from pygrank.measures import Measure, AUC
from pygrank.measures.utils import split
from typing import Callable, Iterable, Tuple
from pygrank.core import backend


default_tuning_optimization = {
    "max_vals": [0.95] * 10,
    "min_vals": [0.5] * 10,
    "deviation_tol": 0.005,
    "parameter_tol": 1,
    "verbose": False,
    "divide_range": "shrinking",
    "partitions": 5,
    "depth": 1
}


class SelfClearDict(dict):
    """
    A dictionary that holds only one entry at all times by clearing itself before item assignment.
    This can be passed as an `optimization_dict` argument to closed form filters to ensure that
    back-to-back calls for the same personalization (e.g. by tuning) do not recompute graph convolutions
    while also erasing past hashed values. To keep past convolutions even after calling algorithms with
    different personalization signals, use a simple dictionary instead.
    """

    def __setitem__(self, key, value):
        self.clear()
        super().__setitem__(key, value)


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


class ParameterTuner(Tuner):
    """
    Tunes a parameterized version of node ranking algorithms under a specific measure by splitting the personalization
    in training and test sets.
    """
    def __init__(self, ranker_generator: Callable[[list], NodeRanking] = None,
                 measure: Callable[[GraphSignal, GraphSignal], Measure] = AUC,
                 fraction_of_training: float = 0.8,
                 combined_prediction: bool = True,
                 tuning_backend: str = None,
                 **kwargs):
        """
        Instantiates the tuning mechanism.
        Args:
            ranker_generator: A callable that constructs a ranker based on a list of parameters.
                If None (default) then a pygrank.algorithms.learnable.GenericGraphFilter
                is constructed with automatic normalization and assuming immutability (this is the most common setting).
                These parameters can be overriden and other ones can be passed to the algorithm'personalization
                constructor by including them in kwargs.
            measure: Callable to constuct a supervised measure with given known node scores and an iterable of excluded
                scores.
            fraction_of_training: A number in (0,1) indicating how to split provided graph signals into training and
                validaton ones by randomly sampling training nodes to meet the required fraction of all graph nodes.
                Numbers outside this range can also be used (not recommended without specific reason) per the
                conventions of `pygrank.split(...)`. Default is 0.8.
            combined_prediction: If True (default), after the best version of algorithms is determined, the whole
                personalization is used to produce the end-result. Otherwise, only the training portion of the
                training-validation split is used.
            tuning_backend: Specifically switches to a designted backend for the tuning process before restoring
                the previous one to perform the actual ranking. If None (default), this functionality is ignored.
            kwargs: Additional arguments can be passed to pygrank.algorithms.autotune.optimization.optimize. Otherwise,
                the respective arguments are retrieved from the variable *default_tuning_optimization*, which is crafted
                for fast convergence of the default ranker_generator. Arguments passable to the ranker_generator are
                also passed to it.
                Make sure to declare both the upper **and** the lower bounds of parameter values.

        Example:
            >>> import pygrank as pg
            >>> graph, personalization = ...
            >>> tuner = pg.ParameterTuner(measure=AUC, deviation_tol=0.01)
            >>> ranks = tuner.rank(graph, personalization)

        Example to tune pagerank'personalization float parameter alpha in the range [0.5, 0.99]:
            >>> import pygrank as pg
            >>> graph, personalization = ...
            >>> tuner = pg.ParameterTuner(lambda params: pg.PageRank(alpha=params[0]),
            >>>                           measure=AUC, deviation_tol=0.01, max_vals=[0.99], min_vals=[0.5])
            >>> ranks = tuner.rank(graph, personalization)
        """
        if ranker_generator is None:
            from pygrank.algorithms import GenericGraphFilter
            if 'preprocessor' not in kwargs and 'assume_immutability' not in kwargs and 'normalization' not in kwargs:
                kwargs['preprocessor'] = preprocessor(assume_immutability=True)
            if "optimization_dict" not in kwargs:
                kwargs["optimization_dict"] = SelfClearDict()

            def ranker_generator(params):
                return GenericGraphFilter(params, **remove_used_args(optimize, kwargs))
        else:
            ensure_used_args(kwargs, [optimize])
        self.ranker_generator = ranker_generator
        self.measure = measure
        self.fraction_of_training = fraction_of_training
        self.optimize_args = {kwarg: kwargs.get(kwarg, val) for kwarg, val in default_tuning_optimization.items()}
        self.combined_prediction = combined_prediction
        self.tuning_backend = tuning_backend

    def _run(self, personalization: GraphSignal, params: list, *args, **kwargs):
        return self.ranker_generator(params).rank(personalization, *args, **kwargs)

    def _tune(self, graph=None, personalization=None, *args, **kwargs):
        previous_backend = backend.backend_name()
        personalization = to_signal(graph, personalization)
        if self.tuning_backend is not None and self.tuning_backend != previous_backend:
            backend.load_backend(self.tuning_backend)
        backend_personalization = to_signal(graph, backend.to_array(personalization.np))
        training, validation = split(backend_personalization, self.fraction_of_training)
        measure = self.measure(validation, training)
        best_params = optimize(
            lambda params: -measure.best_direction()*measure.evaluate(self._run(training, params, *args, **kwargs)),
            **self.optimize_args)
        if self.tuning_backend is not None and self.tuning_backend != previous_backend:
            backend.load_backend(previous_backend)
            # TODO: make training back-propagate through tensorflow for combined_prediction==False (do this with a gather in the split method)
        return self.ranker_generator(best_params), personalization if self.combined_prediction else training

    def references(self):
        desc = "parameters tuned \\cite{krasanakis2021pygrank} to optimize "+self.measure(no_signal, no_signal).__class__.__name__\
               +f" while withholding {1-self.fraction_of_training:.3f} of nodes for validation"
        ret = self.ranker_generator([-42]).references()  # an invalid parameter value
        for i in range(len(ret)):
            if "-42" in ret[i]:
                ret[i] = desc
                return ret
        return ret + [desc]


class AlgorithmSelection(Tuner):
    def __init__(self, rankers: Iterable[NodeRanking] = None,
                 measure: Callable[[GraphSignal, GraphSignal], Measure] = AUC,
                 fraction_of_training: float = 0.8,
                 combined_prediction: bool = True,
                 tuning_backend: str = None):
        """
        Instantiates the tuning mechanism.
        Args:
            rankers: An iterable of node ranking algorithms to chose from. Try to make them share a preprocessor
                for more efficient computations. If None (default), the filters obtained from
                pygrank.benchmark.create_demo_filters().values() are used instead.
            measure: Callable to constuct a supervised measure with given known node scores and an iterable of excluded
                scores.
            fraction_of_training: A number in (0,1) indicating how to split provided graph signals into training and
                validaton ones by randomly sampling training nodes to meet the required fraction of all graph nodes.
                Numbers outside this range can also be used (not recommended without specific reason) per the
                conventions of `pygrank.split(...)`. Default is 0.8.
            combined_prediction: If True (default), after the best version of algorithms is determined, the whole
                personalization is used to produce the end-result. Otherwise, only the training portion of the
                training-validation split is used.
            tuning_backend: Specifically switches to a designated backend for the tuning process before restoring
                the previous one to perform the actual ranking. If None (default), this functionality is ignored.

        Example:
            >>> import pygrank as pg
            >>> graph, personalization = ...
            >>> tuner = pg.AlgorithmSelection(pg.create_demo_filters().values(), measure=pg.AUC, deviation_tol=0.01)
            >>> ranks = tuner.rank(graph, personalization)

        Example (with more filters):
            >>> import pygrank as pg
            >>> graph, personalization = ...
            >>> algorithms = pg.create_variations(pg.create_many_filters(tol=1.E-9), pg.create_many_variation_types())
            >>> tuner = pg.AlgorithmSelection(algorithms.values(), measure=pg.AUC, deviation_tol=0.01)
            >>> ranks = tuner.rank(graph, personalization)
        """
        if rankers is None:
            from pygrank.benchmarks import create_demo_filters
            rankers = create_demo_filters().values()
        self.rankers = rankers
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
            value = measure.best_direction()*measure.evaluate(ranker.rank(training, *args, **kwargs))
            if value > best_value:
                best_value = value
                best_ranker = ranker
        if self.tuning_backend is not None and self.tuning_backend != previous_backend:
            backend.load_backend(previous_backend)
            # TODO: make training back-propagate through tensorflow for combined_prediction==False
        return best_ranker, personalization if self.combined_prediction else training

    def references(self):
        desc = "selected the best among the following algorithms that optimizes "+self.measure(no_signal, no_signal).__class__.__name__ \
               +f" while withholding {1-self.fraction_of_training:.3f} of nodes for validation: \\\\\n"
        for ranker in self.rankers:
            desc += "  - "+ranker.cite()+" \\\\\n"
        return [desc]
