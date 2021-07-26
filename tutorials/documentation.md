<center><h1 style=font-size:200px>Documentation</h1></center> 

# Graph Signals
Graph signals are a way to organize numerical values corresponding to respective
nodes. They
are returned by ranking algorithms, but for ease-of-use you can also pass
maps of node values (e.g.  `{"A": 3, "C": 2}`)
or numpy arrays (e.g. `np.array([3, 0, 2, 0])` where positions correspond
to the order networkx traverse graph nodes) to them. If so,
these representations are converted internally to graph signals based on
whatever graph information is available.

### :zap: Example
As an example, let us create a simple line graph of three edges `"A", "B", "C"` 
and assign to the first and the last one the values *3* and *2* respectively.
To create a graph signal holding this information we can write:

```python
>>> from pygrank.algorithms.utils.graph_signal import to_signal
>>> import networkx as nx
>>> G = nx.Graph()
>>> G.add_edge("A", "B")
>>> G.add_edge("B", "C")
>>> signal = to_signal(G, {"A": 3, "C": 2})
>>> print(signal["A"], signal["B"])
3.0 0.0
```

If is possible to directly access graph signal values as numpy arrays
through a `signal.np` attribute. Continuing from the previous example,
in the following code we divide a graph signal's elements with their sum.
Value changes are reflected to the values being accessed.

```python
>>> print(signal.np)
[3. 0. 2.]
>>> signal.np /= signal.np.sum()
>>> print([(k,v) for k,v in signal.items()])
[('A', 0.6), ('B', 0.0), ('C', 0.4)]
```

### :hammer_and_wrench: Details
For ease of use, the library can parse
dictionaries that map nodes to values, e.g. `{"A":0.8,"B":0.5}`
where ommitted nodes are considered to correspond to zeroes,
or numpy arrays with the same number of elements as graph nodes,
e.g. `np.ndarray([node_scores.get(v, 0) for v in graph])` where `graph` is the networkx
graph passed to node ranking algorithms. When either of these two conventions is used,
ranking algorithms are automatically converted them to graph signals.

At the same time, the output of `rank(...)` methods are always graph signals. It must
be noted that this datatype implements the same methods as a dictionary and can
be used interchangeably, whereas access to a numpy array storing corresponding node
values can be obtained through the object attribute `signal.np`.



# Graph Filters
Graph filters are ways to diffuse graph signals through graphs by sending
node values to their neighbors and aggregating them there. 

### :zap: Example


### :brain: Explanation
The main principle
lies in recognizing that propagating a graph signal's vector (i.e. numpy array)
represntation `p` one hop away in the graph is performed through the operation
`Mp`, where `M` is a normalization of the graph's adjacency matrix. To gain an
intuition, think of column-based normalization, where `Mp`
becomes an update of all node values by setting them as their
neighbors' previous average.

### :hammer_and_wrench: Details
The library provides several graph filters. Their usage pattern consists
of instantiating them and then calling their `rank(graph, personalization)`
method to obtain posterior node signals based on diffusing the provided
personalization signal through the graph.

### :scroll: List of Graph Filters
An exhaustive list of all ready-to-use graph filters can be
found [here](graph_filters.md). After initialization with the appropriate
parameters, these can be used interchangeably in the above example.

# Postprocessors
Postprocessors wrap base graph filters to affect their outcome. Usage
of the original filters remains identical.

### :zap: Example

### :brain: Explanation
There are many ways graph filter posteriors can be processed to provide
more meaningful data. Of the simpler ones are normalization constraints,
for example to set the maximal or the sum of posterior node values to
sum to 1. There also exist thresholding schemes, which can be used
for binary community detections, as well as methods to make node
comparisons non-parameteric by transforming scores to ordinalities.

Some more complex postprocessing mechanisms involve re-running the 
base filters with augmented personalization. This happens both for
seed oversampling postprocessors, which aim to augment node scores
by providing more example nodes, and for fairness-aware posteriors,
which aim to make node scores adhere to some fairness constraint, 
such as disparate impact.

### :scroll: List of Graph Filters
An exhaustive list of all ready-to-use postprocessors can be
found [here](postprocessors.md). After initialization with the appropriate
parameters, these can be used interchangeably in the above example.



# Evaluation

### :zap: Examples

### :brain: Evaluation Measures

### :brain: Benchmarks

### :scroll: List of Benchmarks

### :scroll: List of Evaluation Measures
