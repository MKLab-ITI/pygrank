from typing import Callable, Mapping, Any
from pygrank.core import to_signal, GraphSignal, NodeRanking
from pygrank.measures.utils import split
from pygrank.measures import AUC, Supervised
from timeit import default_timer as time
import collections
import networkx as nx


def supervised_benchmark(algorithms: Mapping[str, NodeRanking],
                         datasets: Any,
                         metric: Callable[[GraphSignal, GraphSignal], Supervised] = AUC,
                         fraction_of_training: float = 0.5,
                         seed: int = 0):
    """
    Compares the outcome of provided algorithms on given datasets using a desired metric.

    Args:
        algorithms: A map from names to node ranking algorithms to compare.
        datasets: A list of datasets to compare the algorithms on. List elements should either be strings or (string, num) tuples
            indicating the dataset name and number of community of interest respectively.
        metric: A method to instantiate a measure type to assess the efficacy of algorithms with.
        fraction_of_training: The fraction of training samples to split on. The rest are used for testing.
        seed: A seed to ensure reproducibility. Default is 0.
    Returns:
        Yields an array of outcomes. Is meant to be used with wrapping methods, such as print_benchmark.
    Example:
        >>> import pygrank as pg
        >>> algorithms = ...
        >>> datasets = ...
        >>> pg.benchmark_print(pg.supervised_benchmark(algorithms, datasets))
    """
    yield [""] + [algorithm for algorithm in algorithms]
    for name, graph, group in datasets:
        multigroup = isinstance(group, collections.abc.Mapping) and not isinstance(group, GraphSignal)
        training, evaluation = split(group, training_samples=fraction_of_training, seed=seed)
        if multigroup:
            training = {group_id: to_signal(graph,{v: 1 for v in group}) for group_id, group in training.items()}
            evaluation = {group_id: to_signal(graph,{v: 1 for v in group}) for group_id, group in evaluation.items()}
        else:
            training, evaluation = to_signal(graph, {v: 1 for v in training}), to_signal(graph, {v: 1 for v in evaluation})
        dataset_results = [name]
        for algorithm in algorithms.values():
            if metric == "time":
                tic = time()
                algorithm.rank(graph, training)
                dataset_results.append(time()-tic)
            else:
                if multigroup:
                    predictions = {group_id: algorithm.rank(graph, group) for group_id, group in training.items()}
                else:
                    predictions = algorithm.rank(graph, training)
                dataset_results.append(metric(evaluation, training)(predictions))
        yield dataset_results


def unsupervised_benchmark(algorithms: Mapping[str, NodeRanking],
                         datasets: Any,
                         metric: Callable[[nx.Graph], Supervised] = AUC,
                         fraction_of_training: float = 0.5,
                         seed: int = 0):
    """
    Compares the outcome of provided algorithms on given datasets using a desired metric.

    Args:
        algorithms: A map from names to node ranking algorithms to compare.
        datasets: A list of datasets to compare the algorithms on. List elements should either be strings or (string, num) tuples
            indicating the dataset name and number of community of interest respectively.
        metric: A method to instantiate a measure type to assess the efficacy of algorithms with.
        fraction_of_training: The fraction of training samples to split on. The rest are used for testing.
        seed: A seed to ensure reproducibility. Default is 0.
    Returns:
        Yields an array of outcomes. Is meant to be used with wrapping methods, such as print_benchmark.
    Example:
        >>> import pygrank as pg
        >>> algorithms = ...
        >>> datasets = ...
        >>> pg.benchmark_print(pg.supervised_benchmark(algorithms, datasets))
    """
    yield [""] + [algorithm for algorithm in algorithms]
    for name, graph, group in datasets:
        multigroup = isinstance(group, collections.abc.Mapping) and not isinstance(group, GraphSignal)
        training, _ = split(group, training_samples=fraction_of_training, seed=seed)
        if multigroup:
            training = {group_id: to_signal(graph,{v: 1 for v in group}) for group_id, group in training.items()}
        else:
            training = to_signal(graph, {v: 1 for v in training})
        dataset_results = [name]
        for algorithm in algorithms.values():
            if multigroup:
                predictions = {group_id: algorithm.rank(graph, group) for group_id, group in training.items()}
            else:
                predictions = algorithm.rank(graph, training)
            dataset_results.append(metric(graph)(predictions))
        yield dataset_results
