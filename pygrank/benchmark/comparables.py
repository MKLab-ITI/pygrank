from typing import Callable, Mapping
from pygrank.algorithms import Postprocessor
from pygrank.algorithms import NodeRanking


def create_variations(algorithms: Mapping[str, NodeRanking], variations: Mapping[str, Callable[[NodeRanking], Postprocessor]]):
    """
    Augments provided algorithms with all possible variations.
    Args:
        algorithms: A map from names to node ranking algorithms to compare.
        variations: A map from names to postprocessor types to wrap around node ranking algorithms.
    Returns:
        A map from names to node ranking algorithms to compare. New names append the variation name.
    """
    all = dict()
    for variation in variations:
        for algorithm in algorithms:
            all[algorithm+variation] = variations[variation](algorithms[algorithm])
    return all
