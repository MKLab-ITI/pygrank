from pygrank.core.signals import GraphSignal, to_signal, NodeRanking, no_signal
from pygrank.measures import Measure, AUC
from pygrank.measures.utils import split
from typing import Callable, Iterable, Union
from pygrank.core import backend
from pygrank.algorithms.autotune.tuning import Tuner
import numpy as np


class AlgorithmSelection(Tuner):
    def __init__(self, rankers: Iterable[NodeRanking] = None,
                 measure: Callable[[GraphSignal, GraphSignal], Measure] = AUC,
                 fraction_of_training: Union[Iterable[float], float] = 0.9,
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
        prev_dropout = kwargs.get("graph_dropout")
        kwargs["graph_dropout"] = 0
        best_value = -float('inf')
        best_ranker = None
        fraction_of_training = self.fraction_of_training if isinstance(self.fraction_of_training, Iterable) else [self.fraction_of_training]
        for ranker in self.rankers:
            values = list()
            for seed, fraction in enumerate(fraction_of_training):
                training, validation = split(backend_personalization, fraction, seed=seed)
                measure = self.measure(validation, training)
                values. append(measure.best_direction()*measure.evaluate(ranker.rank(training, *args, **kwargs)))
            value = np.min(values)
            if value > best_value:
                best_value = value
                best_ranker = ranker
        if self.tuning_backend is not None and self.tuning_backend != previous_backend:
            backend.load_backend(previous_backend)
            # TODO: make training back-propagate through tensorflow for combined_prediction==False
        kwargs["graph_dropout"] = prev_dropout
        return best_ranker, personalization if self.combined_prediction else training

    def references(self):
        desc = "selected the best among the following algorithms \\cite{krasanakis2022autogf} that optimizes "\
               + self.measure(no_signal, no_signal).__class__.__name__ \
               + f" while withholding {1-self.fraction_of_training:.3f} of nodes for validation: \\\\\n"
        for ranker in self.rankers:
            desc += "  - "+ranker.cite()+" \\\\\n"
        return [desc]
