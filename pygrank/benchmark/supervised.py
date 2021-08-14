from typing import Callable, Mapping, Any
from pygrank.core import to_signal, GraphSignal, NodeRanking
from pygrank.measures.utils import split
from pygrank.measures import AUC, Supervised
from timeit import default_timer as time


def supervised_benchmark(algorithms: Mapping[str, NodeRanking],
                         datasets: Any,
                         metric: Callable[[GraphSignal, GraphSignal], Supervised] = AUC,
                         seed: int = 0):
    """
    Compares the outcome of provided algorithms on given datasets using a desired metric.

    Args:
        algorithms: A map from names to node ranking algorithms to compare.
        datasets: A list of datasets to compare the algorithms on. List elements should either be strings or (string, num) tuples
            indicating the dataset name and number of community of interest respectively.
        metric: A method to instantiate a measure type to assess the efficacy of algorithms with.
        seed: A seed to ensure reproducibility. Default is 0.
    Returns:
        Yields an array of outcomes. Is meant to be used with wrapping methods, such as print_benchmark.
    Example:
        >>> import pygrank as pg
        >>> algorithms =
        >>> pg.benchmark_print()
    """
    yield [""] + [algorithm for algorithm in algorithms]
    for name, graph, group in datasets:
        training, evaluation = split(list(group), training_samples=0.5, seed=seed)
        training, evaluation = to_signal(graph, {v: 1 for v in training}), to_signal(graph, {v: 1 for v in evaluation})
        dataset_results = [name]
        for algorithm in algorithms.values():
            if metric == "time":
                tic = time()
                algorithm.rank(graph, training)
                dataset_results.append(time()-tic)
            else:
                dataset_results.append(metric(evaluation, training)(algorithm.rank(graph, training)))
        yield dataset_results
