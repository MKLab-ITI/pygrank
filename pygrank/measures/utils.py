import random
import collections
from pygrank.core import GraphSignal, to_signal, GraphSignalData
from typing import Mapping, Union, Iterable
import networkx as nx


class Measure(object):
    def __call__(self, scores: GraphSignalData):
        return self.evaluate(scores)

    def evaluate(self, scores: GraphSignalData):
        raise Exception("Non-abstract subclasses of Measure should implement an evaluate method")

    def best_direction(self):
        return 1  # TODO: automatically detect unsupervised direction

    def as_supervised_method(self):
        def dummy_constructor(dummy_truth, dummy_exclude=None):
            return self
        return dummy_constructor

    def as_unsupervised_method(self):
        def dummy_constructor(graph=None):
            return self
        return dummy_constructor


def split(groups: Union[GraphSignalData, Mapping[str, GraphSignalData]],
          training_samples: float = 0.8,
          seed: int = 0):
    """
    Splits a graph signal, iterable of map of graph signals and iterables into two same-type objects
    with training and test data respectively. For graph signals, training and test data are
    basically masked to output zeros more times and this method takes care to stratify sampling
    between non-zero and zero values.

    Args:
        groups: The input data to split.
        training_samples: If less than 1, it determines the fraction of training data to use for training. If greater
            than 1, it determines the absolute number of training data points. If 1, the data are not split but
            used for both training and testing. Default is 0.8 to use 80% data for training and the rest 20% for
            testing.
        seed: A sample to introu

    Returns:
        Data with the same organization as the *groups* argument.

    Example:
        >>> import pygrank as pg
        >>> training, test = pg.split(["A", "B", "C", "D"], training_samples=0.5)
    """
    if training_samples == 1:
        return groups, groups
    if isinstance(groups, GraphSignal):
        group = [v for v in groups if groups[v] != 0]
        random.Random(seed).shuffle(group)
        splt = training_samples if training_samples > 1 else int(len(group) * training_samples)
        return to_signal(groups, {v: groups[v] for v in group[:splt]}), to_signal(groups, {v: groups[v] for v in group[splt:]})
    if not isinstance(groups, collections.abc.Mapping):
        group = list(groups)
        random.Random(seed).shuffle(group)
        splt = training_samples if training_samples > 1 else int(len(group) * training_samples)
        return group[:splt], group[splt:]
    testing = {}
    training = {}
    for group_id, group in groups.items():
        training[group_id],testing[group_id] = split(group, training_samples, seed)
    return training, testing


def remove_intra_edges(G: nx.Graph, group: Union[GraphSignalData, Mapping[str, GraphSignalData]]):
    if isinstance(group, collections.abc.Mapping):
        for actual_group in group.values():
            remove_intra_edges(G, actual_group)
    else:
        for v in group:
            for u in group:
                if G.has_edge(v, u) or G.has_edge(u, v):
                    G.remove_edge(v, u)
