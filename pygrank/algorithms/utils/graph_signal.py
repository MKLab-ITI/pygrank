import numpy as np
from collections.abc import MutableMapping


def to_signal(graph, obj):
    known_node2id = None
    if isinstance(graph, GraphSignal):
        known_node2id = graph.node2id
        graph = graph.graph
    elif isinstance(graph, np.ndarray):
        raise Exception("Graph cannot be an array")
    if isinstance(obj, GraphSignal):
        if graph != obj.graph:
            raise Exception("Graph signal tied to a different graph")
        return obj
    return GraphSignal(graph, obj, known_node2id)


class GraphSignal(MutableMapping):
    def __init__(self, graph, obj, node2id=None):
        self.graph = graph
        self.node2id = {v: i for i, v in enumerate(graph)} if node2id is None else node2id
        if isinstance(obj, np.ndarray):
            if len(graph) != len(obj):
                raise Exception("Graph signal arrays should have the same dimensions as graphs")
            self.np = obj
        elif obj is None:
            self.np = np.repeat(1.0, len(graph))
        else:
            self.np = np.repeat(0.0, len(graph))
            for key, value in obj.items():
                self[key] = value

    def __getitem__(self, key):
        return self.np[self.node2id[key]]

    def __setitem__(self, key, value):
        self.np[self.node2id[key]] = float(value)

    def __delitem__(self, key):
        self.np[self.node2id[key]] = 0

    def __iter__(self):
        return iter(self.node2id)

    def __len__(self):
        return len(self.node2id)

    def normalized(self, normalize=True, copy=True):
        if copy:
            return GraphSignal(self.graph, np.copy(self.np), self.node2id).normalized(normalize, copy=False)
        if normalize:
            np_sum = self.np.__abs__().sum()
            if np_sum != 0:
                self.np /= np_sum
        return self


class NodeRanking(object):
    def __call__(self, *args, **kwargs):
        self.rank(*args, **kwargs)

    def rank(self, *args, **kwargs):
        raise Exception("NodeRanking subclasses should implement a rank method")