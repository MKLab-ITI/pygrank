# :zap: Showcase

As a quick start, let us construct a graph 
and a set of nodes. The graph's class can be
imported either from the `networkx` library or from
`pygrank` itself. The two are in large part interoperable
and both can be parsed by our algorithms.
But our implementation is tailored to graph signal
processing needs and thus tends to be faster and consume
only a fraction of the memory.

```python
from pygrank import Graph

graph = Graph()
graph.add_edge("A", "B")
graph.add_edge("B", "C")
graph.add_edge("C", "D")
graph.add_edge("D", "E")
graph.add_edge("A", "C")
graph.add_edge("C", "E")
graph.add_edge("B", "E")
seeds = {"A", "B"}
```

We now run a personalized PageRank [graph filter](documentation/documentation.md#graph-filters)
to score the structural relatedness of graph nodes to the ones of the given set.
First, let us import the library:

```python
import pygrank as pg
```

For instructional purposes,
we experiment with (personalized) *PageRank*. 
Instantiation of this and more filters is described [here](documentation/graph_filters.md),
and can be accessed from the top-level import.
We also set the default values of some parameters: the graph diffusion
rate *alpha* required by this particular filter, a numerical tolerance *tol* at the
convergence point and a graph preprocessing strategy *"auto"* that normalizes
the graph adjacency matrix in either a column-based or symmetric
way, depending on whether the graph is undirected (as in this example)
or not respectively.

```python
ranker = pg.PageRank(alpha=0.85, tol=1.E-6, normalization="auto")
ranks = ranker(graph, {v: 1 for v in seeds})
```

Node ranking outputs are always organized into
[graph signals](documentation/documentation.md#graph-signals).
These can be used like dictionaries for easy access.
For example, printing the scores of some nodes can be done per:

```python
print(ranks["B"], ranks["D"], ranks["E"])
# 0.5173091321819129 0.24969444089457765 0.3415804634807899
```

We alter this outcome so that it outputs node order, 
where higher node scores are assigned lower order,
by wrapping a postprocessor around the base algorithm. 
You can find more postprocessors [here](documentation/postprocessors.md),
including ones to make scores fairness-aware.

```python
ordinals = pg.Ordinals(ranker).rank(graph, {v: 1 for v in seeds})
print(ordinals["B"], ordinals["D"], ordinals["E"])
# 1.0 5.0 4.0
```

How much time did it take for the base ranker to converge?
(Depends on backend and device characteristics.)

```python
print(ranker.convergence)
# 19 iterations (0.0021852000063518062 sec)
```

Since for this example only the node order is important,
we can use a different way to specify convergence:

```python
convergence = pg.RankOrderConvergenceManager(pagerank_alpha=0.85, confidence=0.98) 
early_stop_ranker = pg.PageRank(alpha=0.85, convergence=convergence)
ordinals = pg.Ordinals(early_stop_ranker).rank(graph, {v: 1 for v in seeds})
print(early_stop_ranker.convergence)
# 2 iterations (0.0005241000035312027 sec)
print(ordinals["B"], ordinals["D"], ordinals["E"])
# 3.0 5.0 4.0
```

Close to the previous results at a fraction of the time!! For large graphs,
most ordinals would be near the ideal ones. Note that convergence time 
does not take into account the time needed to preprocess graphs.

Till now, we used `PageRank`, but what would happen if we do not know which base
algorithm to use? In these cases `pygrank` provides online tuning of generalized
graph signal processing filters on the personalization. The ranker
in the ranking algorithm construction code can be replaced with an automatically tuned
equivalent per:

```python
tuned_ranker = pg.ParameterTuner()
ordinals = pg.Ordinals(tuned_ranker).rank(graph, {v: 1 for v in seeds})
print(ordinals["B"], ordinals["D"], ordinals["E"])
# 2.0 5.0 4.0
```

This yields similar node ordinals, which means that tuning constructed
a graph filter similar to `PageRank`.
Tuning may be worse than highly specialized algorithms in some settings, 
but usually finds near-best base algorithms.

To obtain a recommendation about how to cite complex
algorithms, an automated description can be extracted 
by the source code per the following
command:

```python
print(tuned_ranker.cite())
# graph filter \cite{ortega2018graph} with dictionary-based hashing \cite{krasanakis2022pygrank}, max normalization and parameters tuned \cite{krasanakis2022autogf} to optimize AUC while withholding 0.100 of nodes for validation
```
Bibtex entries corresponding to the citations can be found 
[here](documentation/citations.md).