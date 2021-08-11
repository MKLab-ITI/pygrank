from pygrank.algorithms.utils import NodeRanking, preprocessor, to_signal, GraphSignal, _call, _ensure_all_used, _remove_used
from pygrank.algorithms.autotune.optimization import optimize
from pygrank.measures import Supervised, AUC
from pygrank.measures.utils import split
from typing import Callable


default_tuning_optimization = {
    "max_vals": [0.95] * 10,
    "min_vals": [0.5] * 10,
    "deviation_tol": 0.005,
    "parameter_tol": 0.1,
    "verbose": False,
    "divide_range": "shrinking",
    "partitions": 5,
    "depth": 1
}


class Tuner(NodeRanking):
    pass


class ParameterTuner(Tuner):
    """
    Tunes a parameterized version of node ranking algorithms under a specific measure by splitting the personalization
    in training and test sets.
    """
    def __init__(self, ranker_generator : Callable[[list], NodeRanking] = None,
                 measure : Callable[[GraphSignal, GraphSignal], Supervised] = AUC,
                 fraction_of_training : float = 0.5,
                 combined_prediction : bool = True,
                 **kwargs):
        """
        Instantiates the tuning mechanism.
        Args:
            ranker_generator: A callable that constructs a ranker based on a list of parameters.
                If None (default) then a pygrank.algorithms.learnable.GenericGraphFilter
                is constructed with automatic normalization and assuming immutability (this is the most common setting).
                These parameters can be overriden and other ones can be passed to the algorithm's constructor simply
                by including them in kwargs.
            measure: Callable to constuct a supervised measure with given known node scores and an iterable of excluded scores.
            fraction_of_training: A number in (0,1) indicating how to split provided graph signals into training and
                validaton ones by randomly sampling training nodes to meet the required fraction of all graph nodes. Default is 0.5.
            combined_prediction: If True (default), after the best version of algorithms is determined, the whole personalization is used
                to produce the end-result. Otherwise, only the training portion of the training-validation split is used.
            kwargs: Additional arguments can be passed to pygrank.algorithms.autotune.optimization.optimize. Otherwise,
                the respective arguments are retrieved from the variable *default_tuning_optimization*, which is crafted
                for fast convergence of the default ranker_generator.
                Make sure to declare both the upper **and** the lower bounds of parameter values.

        Example:
            >>> from pygrank.algorithms.autotune import ParameterTuner
            >>> graph, personalization = ...
            >>> tuner = ParameterTuner(measure=AUC, deviation_tol=0.01)
            >>> ranks = tuner.rank(graph, personalization)

        Example to tune pagerank's float parameter alpha in the range [0.5, 0.99]:
            >>> from pygrank.algorithms.adhoc import PageRank
            >>> from pygrank.algorithms.autotune import ParameterTuner
            >>> graph, personalization = ...
            >>> tuner = ParameterTuner(lambda params: PageRank(alpha=params[0]), measure=AUC, deviation_tol=0.01, max_vals=[0.99], min_vals=[0.5])
            >>> ranks = algorithm.rank(graph, personalization)
        """
        if ranker_generator is None:
            from pygrank.algorithms.learnable import GenericGraphFilter
            if 'to_scipy' not in kwargs and 'assume_immutability' not in kwargs and 'normalization' not in kwargs:
                kwargs['to_scipy'] = preprocessor(assume_immutability=True)
            ranker_generator = lambda params: GenericGraphFilter(params, **_remove_used(optimize, kwargs))
        else:
            _ensure_all_used(kwargs, [optimize])
        self.ranker_generator = ranker_generator
        self.measure = measure
        self.fraction_of_training = fraction_of_training
        self.optimize_args = {kwarg: kwargs.get(kwarg, val) for kwarg, val in default_tuning_optimization.items()}
        self.combined_prediction = combined_prediction

    def _run(self, personalization: GraphSignal, params: list, *args, **kwargs):
        return self.ranker_generator(params).rank(personalization, *args, **kwargs)

    def _tune(self, graph=None, personalization=None, *args, **kwargs):
        personalization = to_signal(graph, personalization)
        training, validation = split(personalization, self.fraction_of_training)
        params = optimize(
            lambda params: -self.measure(validation, training).evaluate(self._run(training, params, *args, **kwargs)),
            **self.optimize_args)
        return self.ranker_generator(params), training

    def tune(self, graph=None, personalization=None, *args, **kwargs):
        return self._tune(graph, personalization, *args, **kwargs)[0]

    def rank(self, graph=None, personalization=None, *args, **kwargs):
        ranker, training = self._tune(graph, personalization, *args, **kwargs)
        return ranker.rank(graph, personalization if self.combined_prediction else training, *args, **kwargs)


