from pygrank.algorithms.autotune.tuning import Tuner
from pygrank.algorithms.autotune.optimization import nelder_mead, optimize, lbfgsb
from typing import Callable, Optional, Union, Iterable
from pygrank.core import GraphSignal, to_signal, NodeRanking, no_signal, ensure_used_args, remove_used_args
from pygrank.core import preprocessor, backend
from pygrank.measures import Measure, AUC, split


default_tuning_optimization = {
    "max_vals": [1]+[1] * 40,
    "min_vals": [1]+[0] * 40,
    "deviation_tol": 1.E-6,
    "parameter_tol": 1,
    "verbose": True,
    "divide_range": 1.01,
    "partitions": 5,
    "depth": 1,
    "coarse": 0,
    "shrink_strategy": "divide",
    "partition_strategy": "split"
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


class ParameterTuner(Tuner):
    """
    Tunes a parameterized version of node ranking algorithms under a specific measure by splitting the personalization
    in training and test sets.
    """
    def __init__(self, ranker_generator: Callable[[list], NodeRanking] = None,
                 measure: Callable[[GraphSignal, GraphSignal], Measure] = AUC,
                 fraction_of_training: Union[Iterable[float], float] = 0.9,
                 cross_validate: int = 1,
                 combined_prediction: bool = True,
                 tuning_backend: str = None,
                 pre_diffuse: Optional[NodeRanking] = None,
                 optimizer=optimize,
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
                conventions of `pygrank.split(...)`. Default is 0.5.
            cross_validate: Averages the optimal parameters along a specified number of validation splits.
                Default is 1.
            combined_prediction: If True (default), after the best version of algorithms is determined, the whole
                personalization is used to produce the end-result. Otherwise, only the training portion of the
                training-validation split is used.
            tuning_backend: Specifically switches to a designted backend for the tuning process before restoring
                the previous one to perform the actual ranking. If None (default), this functionality is ignored.
            optimizer: The optimizer of choice to use. Default is `pygrank.algorithms.autotune.optimization.optimize`,
                but other methods can be used such as Default is `pygrank.algorithms.autotune.optimization.evolutionary_optimizer`.
                Parameters to the optimizer need to be passed via kwargs.
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
            from pygrank.algorithms import GenericGraphFilter, Normalize
            if 'preprocessor' not in kwargs and 'assume_immutability' not in kwargs and 'normalization' not in kwargs:
                kwargs['preprocessor'] = preprocessor(assume_immutability=True)
            if "optimization_dict" not in kwargs:
                kwargs["optimization_dict"] = SelfClearDict()

            def ranker_generator(params):
                return Normalize(GenericGraphFilter(params, **remove_used_args(optimize, kwargs)))
        #else:
        #    ensure_used_args(kwargs, [optimizer]) # TODO: find how to do this
        self.ranker_generator = ranker_generator
        self.measure = measure
        self.fraction_of_training = fraction_of_training
        self.optimize_args = {kwarg: kwargs.get(kwarg, val) for kwarg, val in default_tuning_optimization.items()}
        self.combined_prediction = combined_prediction
        self.tuning_backend = tuning_backend
        self.pre_diffuse = pre_diffuse
        self.cross_validate = cross_validate
        self.optimizer = optimizer

    def _run(self, personalization: GraphSignal, params: list, *args, **kwargs):
        return self.ranker_generator(params).rank(personalization, *args, **kwargs)

    def _tune(self, graph=None, personalization=None, *args, **kwargs):
        previous_backend = backend.backend_name()
        personalization = to_signal(graph, personalization)
        if self.tuning_backend is not None and self.tuning_backend != previous_backend:
            backend.load_backend(self.tuning_backend)
        backend_personalization = to_signal(graph, backend.to_array(personalization.np))
        total_params = list()
        for seed0 in range(self.cross_validate):
            fraction_of_training = self.fraction_of_training if isinstance(self.fraction_of_training, Iterable) else [ self.fraction_of_training]
            #fraction_of_training = [random.choice(fraction_of_training)]
            internal_training_list = list()
            validation_list = list()
            for seed, fraction in enumerate(fraction_of_training):
                training, validation = split(backend_personalization, fraction, seed0+seed)
                internal_training = training
                if self.pre_diffuse is not None:
                    internal_training = self.pre_diffuse(internal_training)
                    validation = self.pre_diffuse(validation)
                internal_training_list.append(internal_training)
                validation_list.append(validation)

            def eval(params):
                val = 0
                for internal_training, validation in zip(internal_training_list, validation_list):
                    """import pygrank as pg

                    scores = self._run(backend_personalization, params, *args, **kwargs)
                    internal_training = pg.Undersample(int(backend.sum(internal_training)))(scores*backend_personalization)
                    validation = backend_personalization - internal_training"""
                    measure = self.measure(validation, internal_training if internal_training != validation else None)
                    val = val-measure.best_direction() * measure.evaluate(self._run(internal_training, params, *args, **kwargs))
                return val / len(internal_training_list)

            best_params = self.optimizer(eval, **self.optimize_args)
            """import cma
            es = cma.CMAEvolutionStrategy([0.5 for _ in range(len(self.optimize_args["max_vals"]))], 1./12**0.5)
            es.optimize(eval, verb_disp=False)
            best_params = es.result.xbest"""
            total_params.append(best_params)
        best_params = [0 for _ in best_params]
        best_squares = [0 for _ in best_params]
        best_means = [0 for _ in best_params]
        for params in total_params:
            for i in range(len(best_params)):
                best_params[i] = max(best_params[i], params[i])
                best_means[i] += params[i]/self.cross_validate
                best_squares[i] += params[i]**2/self.cross_validate
        #best_params = best_means
        #print(best_params)
        #print(eval(best_params))
        #print("means", best_means)
        #print("stds", [(best_squares[i]-best_means[i]**2)**0.5 for i in range(len(best_params))])
        best_params = best_means
        #if self.cross_validate > 1:
        #    best_params = self.optimizer(eval, **self.optimize_args, weights=best_params)

        if self.tuning_backend is not None and self.tuning_backend != previous_backend:
            backend.load_backend(previous_backend)
            # TODO: make training back-propagate through tensorflow for combined_prediction==False (do this with a gather in the split method)
        self.last_params = best_params
        #print(best_params)
        return self.ranker_generator(best_params), personalization if self.combined_prediction else internal_training

    def references(self):
        desc = "parameters tuned \\cite{krasanakis2022autogf} to optimize "+self.measure(no_signal, no_signal).__class__.__name__\
               + f" while withholding {1-self.fraction_of_training:.3f} of nodes for validation"
        ret = self.ranker_generator([-42]).references()  # an invalid parameter value
        for i in range(len(ret)):
            if "-42" in ret[i]:
                ret[i] = desc
                return ret
        return ret + [desc]
