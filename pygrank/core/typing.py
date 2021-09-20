from typing import Union, Optional, Iterable, Mapping, List
import networkx as nx
import numpy as np


BackendPrimitive = Union["Tensor", np.ndarray, float, List[float]]
BackendGraph = Union["Tensor", np.ndarray]
GraphSignalGraph = Optional[Union[nx.Graph, "GraphSignal"]]
GraphSignalData = Optional[Union["GraphSignal", BackendPrimitive, Iterable[float], Mapping[object, float]]]
