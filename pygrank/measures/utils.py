import random
import collections
from pygrank.algorithms.utils import GraphSignal, to_signal


class Measure(object):
    def __call__(self, ranks):
        return self.evaluate(ranks)

    def evaluate(self, ranks):
        raise Exception("Non-abstract subclasses of Measure should implement an evaluate method")


def to_nodes(groups):
    if not isinstance(groups, collections.Mapping) or isinstance(groups, GraphSignal):
        return list(set(groups))
    all_nodes = list()
    for group in groups.values():
        all_nodes.extend(group)
    return list(set(all_nodes))


def split(groups, training_samples=0.8):
    if training_samples == 1:
        return groups, groups
    if isinstance(groups, GraphSignal):
        group = list(groups)
        random.shuffle(group)
        splt = training_samples if training_samples > 1 else int(len(group) * training_samples)
        return to_signal(groups, {v: groups[v] for v in group[:splt]}), to_signal(groups, {v: groups[v] for v in group[splt:]})
    if not isinstance(groups, collections.Mapping):
        group = list(groups)
        random.shuffle(group)
        splt = training_samples if training_samples > 1 else int(len(group) * training_samples)
        return group[:splt], group[splt:]
    clusters = {}
    training = {}
    for group_id, group in groups.items():
        splt = training_samples if training_samples > 1 else int(len(group) * training_samples)
        if splt < 1:
            splt = 1
        # group = list(group) # not really needed if data are already imported as lists
        random.shuffle(group)
        training[group_id] = group[:splt]
        clusters[group_id] = group[splt:]
    return training, clusters


def remove_group_edges_from_graph(G, group):
    if isinstance(group, collections.Mapping):
        for actual_group in group.values():
            remove_group_edges_from_graph(G, actual_group)
    else:
        for v in group:
            for u in group:
                if G.has_edge(v,u):
                    G.remove_edge(v,u)
                if G.has_edge(u, v):
                    G.remove_edge(u,v)
