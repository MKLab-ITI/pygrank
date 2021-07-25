![pygrank](tutorials/pygrank.png)

Fast recommendation algorithms for large graphs based on link analysis.
<br>
<br>

**License:** Apache Software License
<br>**Author:** Emmanouil (Manios) Krasanakis
<br>**Dependencies:** `tqdm, sklearn, scipy, numpy, networkx`

# :hammer_and_wrench: Installation
`pygrank` is meant to work with Python 3.6 or later. It can be installed with pip per:
```
pip install pygrank
```

# :zap: Quickstart
As a quick start, let us construct a networkx graph `G` and a set of nodes `seeds`.

```python
>>> import networkx as nx
>>> G = nx.Graph()
>>> G.add_edge("A", "B")
>>> G.add_edge("B", "C")
>>> G.add_edge("C", "D")
>>> G.add_edge("D", "E")
>>> G.add_edge("A", "C")
>>> G.add_edge("C", "E")
>>> G.add_edge("B", "E")
>>> seeds = {"A", "B"}
```

We run a personalized PageRank [graph filter](tutorials/documentation.md#graph-filters)
to score the structural relatedness
of graph nodes to the ones of the given set. For instructional purposes,
we show the default values of some parameters: the graph diffusion
rate *alpha* required by the algorithm, a numerical tolerance *tol* at the
convergence point and a graph preprocessing strategy *"auto"* normalization
of the garph adjacency matrix to determine between column-based and symmetric
normalization depending on whether the graph is undirected (as in this example)
or not respectively.
```python
>>> from pygrank.algorithms.adhoc import PageRank
>>> ranker = PageRank(alpha=0.85, tol=1.E-6, normalization="auto")
>>> ranks = ranker.rank(G, {v: 1 for v in seeds})
```

Node ranking output is always organized into
[graph signals](tutorials/documentation.md#graph-signals)
which can be used like dictionaries. For example, we can
print the scores of some nodes per:
```python
>>> print(ranks["B"], ranks["D"], ranks["E"])
0.19245466859447746 0.14087481834802176 0.17014304812714195
```

We alter this outcome so that it outputs node order, 
where higher node scores are assigned lower order. This is achieved
by wrapping a postprocessor around the algorithm. There are various
postprocessors, including ones to make scores fairness-aware.

```python
>>> from pygrank.algorithms.postprocess import Ordinals
>>> ordinals = Ordinals(ranker).rank(G, {v: 1 for v in seeds})
>>> print(ordinals["B"], ordinals["D"], ordinals["E"])
5 1 3
```

How much time did it take for the base ranker to converge?

```python
>>> print(ranker.convergence)
19 iterations (0.0018321000000014465 sec)
```

Since only the node order is important,
we can use a different way to specify convergence:

```python
>>> from pygrank.algorithms.utils.convergence import RankOrderConvergenceManager
>>> convergence = RankOrderConvergenceManager(pagerank_alpha=0.85, confidence=0.98) 
>>> early_stop_ranker = PageRank(alpha=0.85, convergence=convergence)
>>> ordinals = Ordinals(early_stop_ranker).rank(G, {v: 1 for v in seeds})
>>> print(early_stop_ranker.convergence)
3 iterations (0.0015069 sec)
>>> print(ordinals["B"], ordinals["D"], ordinals["E"])
5 1 2
```

Close to the previous results at a fraction of the iterations!!
For small graphs, like this example, time gains by fewer iterations 
are small. But, for graphs with many nodes, running time is dominated
by and increases linearly with the number of iterations.


# :brain: Overview
Analyzing graph edges (links) between nodes can help rank/score
graph nodes based on their structural proximity to structural
or attribute-based communities of nodes.
With the introduction of graph signal processing and
[decoupled graph neural networks]() the importance of node ranking has drastically 
increased, as its ability to perform inductive learning by quickly
spreading node information through edges has been theoretically and experimentally
corroborated. For example, it can be used to make predictions based on few known
node attributes or base predictions outputted by low-quality feature-based machine
learning models.

`pygrank` is a collection of node ranking algorithms and practices that 
support real-world conditions, such as large graphs and heterogeneous
preprocessing and postprocessing requiremenets. Thus, it provides
ready-to-use tools that simplify deployment of theoretical advancements
and testing of new algorithms.

Some of the library's advantages are:
1. **Compatibility** with [networkx](https://github.com/networkx/networkx).
2. **Datacentric** programming interfaces that do not require data transformations.
3. **Fast** computations; running times scale near-linearly with the number of edges.
4. **Large** graph support; memory requirements scale linearly with the number of edges.
5. **Seamless** pipelines, from graph preprocessing up to evaluation of algorithms.


# :link: Material
[Documentation](tutorials/documentation.md)<br>

# :fire: Features
* Graph filters
* Community detection
* Graph normalization
* Convergence criteria
* Postprocessing (e.g. fairness awareness)
* Evaluation measures
* Benchmarks

# :thumbsup: Contributing
Feel free to contribute in any way, for example through the [issue tracker]() or by participating in [discussions]().
Please check out the [contribution guidelines](tutorials/contributing.md) if you want to bring modifications to the code base.
 
# :notebook: Citation
If `pygrank` has been useful in your research and you would like to cite it in a scientific publication, please refer to the following paper:
```
TBD
```
To publish research that uses provided methods, please cite the appropriate paper(s) listed [here](tutorials/citations.md).