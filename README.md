<center><h1>pygrank</h1></center>  
Fast recommendation algorithms for large graphs based on link analysis.

**License:** Apache Software License

**Author:** Emmanouil Krasanakis

## :hammer_and_wrench: Installation
`pygrank` is meant to work with Python 3.6 or above. It can be installed with pip per:
```
pip install pygrank
```
  
## :zap: Quickstart
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

We can run a personalized PageRank algorithm to find structural relatedness scores
of all graph nodes to the given set.
```python
>>> from pygrank.algorithms.pagerank import PageRank
>>> ranker = PageRank(alpha=0.85, tol=1.E-6)
>>> ranks = ranker.rank(G, {v: 1 for v in seeds})
>>> print(ranks["B"], ranks["D"], ranks["E"])
0.19245466859447746 0.14087481834802176 0.17014304812714195
```

We alter this outcome so that it outputs node order, 
where higher node scores are assigned lower order. This is achieved
by wrapping a postprocessor around the algorithm.

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

Close to the previous one
(slight inaccuracy due to small graph) at
a fraction of the time!!


## :brain: Overview


## :link: Links
[Documentation](old_README.md)

## :fire: Features
* Compatibility with networkx
* Graph filters
* Community detection
* Supports large graphs
* Fast computations
* Graph normalization
* Convergence criteria
* Post-processing (e.g. fairness awareness)
* Structural and supervised evaluation measures

## :thumbsup: Contributing
Feel free to contribute in any way, for example through the [issue tracker]() or by participating in [discussions]().
Please check out the [contribution guidelines] if you want to bring modifications to the code base.
 
## :notebook: Citation
If `pygrank` has been useful in your research and you would like to cite it in a scientific publication, please refer to the following paper:
```
TBD
```
To publish research that uses provided methods, please cite the appropriate paper(s) listed [here](tutorials/citations.md).