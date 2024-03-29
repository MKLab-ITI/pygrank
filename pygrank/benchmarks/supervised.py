from typing import Callable, Mapping, Any

from pygrank import Utility
from pygrank.core import to_signal, GraphSignal, NodeRanking
from pygrank.measures.utils import split
from pygrank.measures import AUC, Measure, Time, Unsupervised
from timeit import default_timer as time
from typing import Union, List, Iterable, Optional
import networkx as nx
import collections
from inspect import isclass


def benchmark(algorithms: Mapping[str, NodeRanking],
              datasets: Any,
              metrics: Union[Union[Callable[[nx.Graph], Measure], Callable[[GraphSignal, GraphSignal], Measure]], List[Union[Callable[[nx.Graph], Measure], Callable[[GraphSignal, GraphSignal], Measure]]]] = AUC,
              fraction_of_training: Union[float, Iterable[float]] = 0.5,
              sensitive: Optional[Union[Callable[[nx.Graph], Measure], Callable[[GraphSignal, GraphSignal], Measure]]] = None,
              seed: Union[int, Iterable[int]] = 0):
    """
    Compares the outcome of provided algorithms on given datasets using a desired metric.

    Args:
        algorithms: A map from names to node ranking algorithms to compare.
        datasets: A list of datasets to compare the algorithms on. List elements should either be strings or (string, num) tuples
            indicating the dataset name and number of community of interest respectively.
        metric: A method to instantiate a measure type to assess the efficacy of algorithms with.
        fraction_of_training: The fraction of training samples to split on. The rest are used for testing. An
            iterable of floats can also be provided to experiment with multiple fractions.
        sensitive: Optinal. A generator of sensitivity-aware supervised or unsupervised measures.
            Could be None (default).
        seed: A seed to ensure reproducibility. Default is 0. An iterable of floats can also be provided to experimet
            with multiple seeds.
    Returns:
        Yields an array of outcomes. Is meant to be used with wrapping methods, such as print_benchmark.
    Example:
        >>> import pygrank as pg
        >>> algorithms = ...
        >>> datasets = ...
        >>> pg.benchmark_print(pg.benchmark(algorithms, datasets))
    """
    if not isinstance(metrics, list):
        metrics = [metrics]
    if sensitive is not None or len(metrics) > 1:
        yield [""] + [algorithm for algorithm in algorithms for suffix in [metric.__name__ for metric in metrics]+[sensitive.__name__]]
        yield [""] + [suffix for algorithm in algorithms for suffix in [metric.__name__ for metric in metrics]+[sensitive.__name__]]
    elif len(metrics) > 1:
        yield [""] + [algorithm for algorithm in algorithms for suffix in [metric.__name__ for metric in metrics]]
        yield [""] + [suffix for algorithm in algorithms for suffix in [metric.__name__ for metric in metrics]]
    else:
        yield [""] + [algorithm for algorithm in algorithms]
    seeds = [seed] if isinstance(seed, int) else seed
    fraction_of_training = [fraction_of_training] if isinstance(fraction_of_training, float) or isinstance(fraction_of_training, int) else fraction_of_training
    for name, graph, group in datasets:
        for training_samples in fraction_of_training:
            for seed in seeds:
                multigroup = isinstance(group, collections.abc.Mapping) and not isinstance(group, GraphSignal)
                training, evaluation = split(group, training_samples=training_samples, seed=seed)
                if sensitive is None and multigroup:
                    training = {group_id: to_signal(graph,{v: 1 for v in group}) for group_id, group in training.items()}
                    evaluation = {group_id: to_signal(graph,{v: 1 for v in group}) for group_id, group in evaluation.items()}
                    rank = lambda algorithm: {group_id: algorithm(graph, group) for group_id, group in training.items()}
                else:
                    if multigroup:
                        training = training[0]
                        evaluation = evaluation[0]
                        sensitive_signal = to_signal(graph, {v: 1 for v in group[max(group.keys())]})
                        training, evaluation = to_signal(graph, {v: 1 for v in training}), to_signal(graph, {v: 1 for v in evaluation})
                    else:
                        training, evaluation = to_signal(graph, {v: 1 for v in training}), to_signal(graph, {v: 1 for v in evaluation})
                    if sensitive is not None:
                        if not multigroup:
                            sensitive_signal = to_signal(training, 1-evaluation.np)
                        #training.np = training.np*(1-sensitive_signal.np)
                        rank = lambda algorithm: algorithm(training, sensitive=sensitive_signal)
                    else:
                        rank = lambda algorithm: algorithm(training)
                dataset_results = [name]
                for algorithm in algorithms.values():
                    for metric in metrics:
                        if metric == Time:
                            tic = time()
                            predictions = rank(algorithm)
                            dataset_results.append(time()-tic)
                        else:
                            predictions = rank(algorithm)
                            if (hasattr(metric, "__code__") and metric.__code__.co_argcount == 1) or (isclass(metric) and issubclass(metric, Unsupervised)):
                                dataset_results.append(metric(graph)(predictions))
                            else:
                                dataset_results.append(metric(evaluation, training if training_samples != 1 else None)(predictions))
                    if sensitive is not None:
                        try:
                            dataset_results.append(sensitive(sensitive_signal, training if training_samples != 1 else None)(predictions))
                        except:
                            dataset_results.append(sensitive(evaluation, sensitive_signal, training if training_samples != 1 else None)(predictions))
                yield dataset_results
