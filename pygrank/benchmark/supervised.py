from typing import Callable, Mapping, Any
from pygrank.core import to_signal, GraphSignal, NodeRanking
from pygrank.measures.utils import split
from pygrank.measures import AUC, Supervised


def _perc(num):
    """
    Helper method to pretty print percentages.
    Args:
        num: A number in the range [0,1].
    """
    if num<0.005:
        return "0"
    ret = str(int(num*100+.5)/100.)
    if len(ret) < 4:
        ret += "0"
    if ret[0] == "0":
        return ret[1:]
    return ret


def fill(text="", tab=14):
    """
    Helper method to customly align texts by adding trailing spaces up to a fixed point.
    Args:
        text: The text to add spaces to.
        tab: The alignment point. Default is 14.
    Example:
        >>> print(fill("Text11")+fill("Text12"))
        >>> print(fill("Text21")+fill("Text22"))
    """
    return text+(" "*(tab-len(text)))


def supervised_benchmark(algorithms: Mapping[str, NodeRanking],
                         datasets: Any,
                         metric: Callable[[GraphSignal, GraphSignal], Supervised] = AUC,
                         delimiter: str = " \t ",
                         endline: str = "",
                         verbose: bool = False):
    """
    Compares the outcome of provided algorithms on given datasets using a desired metric.
    Args:
        algorithms: A map from names to node ranking algorithms to compare.
        datasets: A list of datasets to compare the algorithms on. List elements should either be strings or (string, num) tuples
            indicating the dataset name and number of community of interest respectively.
        metric: A method to instantiate a measure type to assess the efficacy of algorithms with.
        delimiter: How to separate columns. Use " & " when exporting to latex format.
        endline: What to print before the end of line. Use "\\\\" when exporting to latex format.
        verbose: Whether to print intermediate steps. Default is False.
    Returns:
        A printable string.
    """
    out = ""
    if verbose:
        print(delimiter.join([fill()]+[fill(algorithm) for algorithm in algorithms])+endline)
    out += delimiter.join([fill()]+[fill(algorithm) for algorithm in algorithms])+endline+"\n"
    for name, graph, group in datasets:
        dataset_results = fill(name)
        training, evaluation = split(list(group), training_samples=0.1)
        training, evaluation = to_signal(graph,{v: 1 for v in training}), to_signal(graph,{v: 1 for v in evaluation})
        for algorithm in algorithms.values():
            dataset_results += delimiter+fill(_perc(metric(evaluation, training)(algorithm.rank(graph, training))))
        if verbose:
            print(dataset_results+endline)
        out += dataset_results+endline+"\n"
    return out
