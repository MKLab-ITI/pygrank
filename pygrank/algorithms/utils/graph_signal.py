import numpy as np
from collections.abc import MutableMapping


def to_signal(graph, obj):
    """
    Converts an object to a GraphSignal tied to an explicit or implicit reference to a graph. This method helps
    convert various ways of expressing graph signals to the same format that algorithms can work with. Prefer
    using GraphSignal instances when developing new code, because these combine the advantages of using hashmaps
    for accessing values with the speed provided by numpy arrrays.

    Args:
        graph: Either a graph or a GraphSignal, where in the second case it takes the value of the latter's graph.
            Prefer using a GraphSignal as reference, as this copies the latter's node2id property without additional
            memory or computations.
        obj: Either a numpy array or a hashmap between graph nodes and their values, in which cases the appropriate
            GraphSignal contructor is called, or a GraphSignal in which case it is also returned and a check is
            performed that these are signals on the same graph. If None, this argument induces a graph signal
            of ones.
    """
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
    """
    Represents a graph signal that assigns numeric values to the nodes of a graph.
    Graph signals should **ALWAYS** be instantiated through the method
    pygrank.algorithms.utils.graph_signal.to_signal(graph, obj) .
    Subclasses a MutableMapping and hence can be accessed as a dictionary.

    Attributes:
        graph: Explicit reference to the graph object the signal is tied to.
        np: A numpy array holding a vector representation of the signal. Editing this also edits node values.
        node2id: A map from graph nodes to their position inside the above-described numpy array.

    Example:
        >>> from pygrank.algorithms.utils.graph_signal import to_signal
        >>> import networkx as nx
        >>> G = nx.Graph()
        >>> G.add_edge("A", "B")
        >>> G.add_edge("B", "C")
        >>> signal = to_signal(G, {"A": 3, "C": 2})
        >>> print(signal["A"], signal["B"])
        3.0 0.0
        >>> print(signal.np)
        [3. 0. 2.]
        >>> signal.np /= signal.np.sum()
        >>> print([(k,v) for k,v in signal.items()])
        [('A', 0.6), ('B', 0.0), ('C', 0.4)]
    """

    def __init__(self, graph, obj, node2id=None):
        """Should **ALWAYS** instantiate graph signals with the method to_signal,
        which handles non-instantiation semantics."""

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
        """
        Copies and the signal into a normalized one, which is subsequently returned.

        Args:
            normalized: If True (default) applies L1 normalization to the values of the copied signal.
            copy: If True (default) a new copy is created, otherwise in-place normalization is performed (if at all)
                and self is returned.
        """
        if copy:
            return GraphSignal(self.graph, np.copy(self.np), self.node2id).normalized(normalize, copy=False)
        if normalize:
            np_sum = self.np.__abs__().sum()
            if np_sum != 0:
                self.np /= np_sum
        return self


class NodeRanking(object):
    """
    A generic node ranking algorithm interfact that effectively transforms GraphSignals.
    Ranking algorithms and postprocessors should subclass this interface and implement
    an appropriate rank method. NodeRanking objects can be used as callables and their arguments
    are passed to their rank methods.
    """

    def __call__(self, *args, **kwargs):
        self.rank(*args, **kwargs)

    def rank(self, *args, **kwargs):
        raise Exception("NodeRanking subclasses should implement a rank method")