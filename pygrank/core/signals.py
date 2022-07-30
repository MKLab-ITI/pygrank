from collections.abc import MutableMapping

import networkx as nx

from pygrank.core import backend
from pygrank.core.typing import GraphSignalGraph, GraphSignalData
from typing import Optional, Mapping


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
        >>> import pygrank as pg
        >>> import networkx as nx
        >>> graph = nx.Graph()
        >>> graph.add_edge("A", "B")
        >>> graph.add_edge("B", "C")
        >>> signal = pg.to_signal(graph, {"A": 3, "C": 2})
        >>> print(signal["A"], signal["B"])
        3.0 0.0
        >>> print(signal.np)
        [3. 0. 2.]
        >>> signal.np /= pg.sum(signal.np)
        >>> print([(k,v) for k,v in signal.items()])
        [('A', 0.6), ('B', 0.0), ('C', 0.4)]
    """

    def __init__(self, graph: GraphSignalGraph, obj: GraphSignalData, node2id: Optional[Mapping[object, int]] = None):
        """Should **ALWAYS** instantiate graph signals with the method to_signal,
        which handles non-instantiation semantics."""

        self.graph = graph
        self.node2id = {v: i for i, v in enumerate(graph)} if node2id is None else node2id
        if backend.is_array(obj):
            if backend.length(graph) != backend.length(obj):
                raise Exception("Graph signal array dimensions " + str(backend.length(obj)) +
                                " should be equal to graph nodes " + str(len(graph)))
            self._np = backend.to_array(obj)
        elif obj is None:
            self._np = backend.repeat(1.0, len(graph))
        else:
            import numpy as np
            self._np = np.repeat(0.0, len(graph))  # tensorflow does not initialize editing of eager tensors
            for key, value in obj.items():
                self[key] = value
            self._np = backend.to_array(self._np)  # make all operations with numpy and then potentially switch to tensorflow
        #if len(self.graph) != backend.length(self._np) or len(self.graph) != len(self.node2id):
        #    raise Exception("Graph signal arrays should have the same dimensions as graphs")

    def filter(self, exclude=None):
        if exclude is not None:
            #exclude = set([key for key, value in to_signal(self, exclude).items() if value != 0])
            #ret = backend.to_array([self[key] for key in self.graph if key not in exclude])
            exclude = to_signal(self, exclude)
            return backend.filter_out(self._np, exclude._np)
        return self._np

    def __rshift__(self, other):
        return other(self)

    @property
    def np(self):
        return backend.to_array(self._np)

    @np.setter
    def np(self, value):
        self._np = backend.to_array(self.__compliant_value(value))

    def __getitem__(self, key):
        return float(self._np[self.node2id[key]])

    def __setitem__(self, key, value):
        self._np[self.node2id[key]] = float(value)

    def __delitem__(self, key):
        self._np[self.node2id[key]] = 0

    def __iter__(self):
        return iter(self.node2id)

    def __len__(self):
        return len(self.node2id)

    def __compliant_value(self, other):
        if isinstance(other, GraphSignal):
            if other.graph != self.graph:
                raise Exception("Can not operate between graph signals of different graphs")
            return other.np
        return other

    def __add__(self, other):
        return GraphSignal(self.graph, self.np + self.__compliant_value(other), self.node2id)

    def __iadd__(self, other):
        self.np = self.np + self.__compliant_value(other)
        return self

    def __radd__(self, other):
        return GraphSignal(self.graph, self.__compliant_value(other) + self.np, self.node2id)

    def __sub__(self, other):
        return GraphSignal(self.graph, self.np - self.__compliant_value(other), self.node2id)

    def __isub__(self, other):
        self.np = self.np - self.__compliant_value(other)
        return self

    def __rsub__(self, other):
        return GraphSignal(self.graph, self.__compliant_value(other) - self.np, self.node2id)

    def __mul__(self, other):
        return GraphSignal(self.graph, self.np * self.__compliant_value(other), self.node2id)

    def __imul__(self, other):
        self.np = self.np * self.__compliant_value(other)
        return self

    def __rmul__(self, other):
        return GraphSignal(self.graph, self.__compliant_value(other) * self.np, self.node2id)

    def __pow__(self, other):
        return GraphSignal(self.graph, self.np ** self.__compliant_value(other), self.node2id)

    def __ipow__(self, other):
        self.np = self.np ** self.__compliant_value(other)
        return self

    def __rpow__(self, other):
        return GraphSignal(self.graph, self.__compliant_value(other) ** self.np, self.node2id)

    def __truediv__(self, other):
        return GraphSignal(self.graph, self.np / self.__compliant_value(other), self.node2id)

    def __itruediv__(self, other):
        self.np = self.np / self.__compliant_value(other)
        return self

    #def __floordiv__(self, other):
    #    return GraphSignal(self.graph, self.np // self.__compliant_value(other), self.node2id)

    #def __ifloordiv__(self, other):
    #    self.np = self.np // self.__compliant_value(other)
    #    return self

    def __rtruediv__(self, other):
        return GraphSignal(self.graph, self.__compliant_value(other) / self.np, self.node2id)

    #def __rfloordiv__(self, other):
    #    return GraphSignal(self.graph, self.__compliant_value(other) // self.np, self.node2id)

    def __neg__(self):
        return GraphSignal(self.graph, -self.np, self.node2id)

    def __pos__(self):
        return self

    def normalized(self, normalize=True, copy=True):
        """
        Copies and the signal into a normalized one, which is subsequently returned.

        Args:
            normalize: If True (default) applies L1 normalization to the values of the copied signal.
            copy: If True (default) a new copy is created, otherwise in-place normalization is performed (if at all)
                and self is returned.
        """
        if copy:
            return GraphSignal(self.graph, backend.copy(self._np), self.node2id).normalized(normalize, copy=False)
        if normalize:
            self._np = backend.self_normalize(self._np)
        return self


class NodeRanking(object):
    """
    A generic node ranking algorithm interface that effectively transforms GraphSignals.
    Ranking algorithms and postprocessors should subclass this interface and implement
    an appropriate rank method. NodeRanking objects can be used as callables and their arguments
    are passed to their rank methods.
    """

    def __call__(self,
                 graph: GraphSignalGraph = None,
                 personalization: GraphSignalData = None,
                 *args, **kwargs) -> GraphSignal:
        return self.rank(graph, personalization, *args, **kwargs)

    def __or__(self, data) -> GraphSignal:
        if not isinstance(data, GraphSignal):
            raise Exception("Can only apply signals into rankers (use pygrank.to_signal(graph, data)) to create those)")
        return self(data)

    def __rshift__(self, other):
        other.__lshift__(self)
        return other

    def rank(self,
             graph: GraphSignalGraph = None,
             personalization: GraphSignalData = None,
             *args, **kwargs) -> GraphSignal:
        raise Exception("NodeRanking subclasses should implement a rank method")

    def propagate(self, graph, features, *args, **kwargs):
        return backend.combine_cols([self.rank(graph, col, *args, **kwargs)._np for col in backend.separate_cols(features)])

    def references(self):
        return ["unknown node ranking algorithm"]

    def cite(self):
        refs = self.references()
        ret = refs[0]
        if len(refs) > 1:
            ret += " with "
            ret += ", ".join(refs[1:-1])
            if len(refs) > 2:
                ret += " and "
            ret += refs[-1]
        return ret


def to_signal(graph: GraphSignalGraph, obj: GraphSignalData) -> GraphSignal:
    """
    Converts an object to a GraphSignal tied to an explicit or implicit reference to a graph. This method helps
    convert various ways of expressing graph signals to the same format that algorithms can work with. Prefer
    using GraphSignal instances when developing new code, because these combine the advantages of using hashmaps
    for accessing values with the speed provided by numpy arrays.

    Args:
        graph: Either a graph or a GraphSignal, where in the second case it takes the value of the personalization's graph.
            Prefer using a GraphSignal as reference, as this copies the personalization's node2id property without additional
            memory or computations. If the graph is None, the second argument needs to be a GraphSignal.
        obj: Either a numpy array or a hashmap between graph nodes and their values, in which cases the appropriate
            GraphSignal contructor is called, or a GraphSignal in which case it is also returned and a check is
            performed that these are signals on the same graph. If None, this argument induces a graph signal
            of ones.
    """
    known_node2id = None
    if obj is None and isinstance(graph, GraphSignal):
        obj, graph = graph, obj
    if graph is None:
        if isinstance(obj, GraphSignal):
            graph = obj.graph
        else:
            raise Exception("None graph allowed only for explicit graph signal input")
    elif isinstance(graph, GraphSignal):
        known_node2id = graph.node2id
        graph = graph.graph
    elif backend.is_array(graph):
        raise Exception("Graph cannot be an array")
    if isinstance(obj, list) and len(obj) != len(graph):
        obj = {v: 1 for v in obj}
    if isinstance(obj, GraphSignal):
        if graph != obj.graph:
            raise Exception("Graph signal tied to a different graph")
        return obj
    return GraphSignal(graph, obj, known_node2id)


no_signal = GraphSignal(nx.Graph(), list())
