import numpy as np
from collections.abc import MutableMapping


def to_signal(graph, obj):
    known_node2id = None
    if isinstance(graph, GraphSignal):
        graph = graph.graph
        known_node2id = graph.node2id
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

    def normalized(self, normalize=True):
        if normalize:
            self.np /= self.np.sum()
        return self
