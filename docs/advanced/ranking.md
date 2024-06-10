# Node Ranking

Here we will see how an appropriate convergence manager
can be used to speed up a node ranking process, where
nodes obtain ordinal values 1,2,3,... based on their
importance in the graph structure (1 is the most 
important node). For starters, let us construct some data to test with:

```python
import pygrank as pg

graph = pg.Graph()
graph.add_edge("A", "B")
graph.add_edge("B", "C")
graph.add_edge("C", "D")
graph.add_edge("D", "E")
graph.add_edge("A", "C")
graph.add_edge("C", "E")
graph.add_edge("B", "E")
seeds = {"A", "B"}
```

We now run a personalized PageRank
to score the structural relatedness of graph nodes to the ones of the given set
and apply a postprocessor that ranks nodes based on their score:

```python
ranker = pg.PageRank(alpha=0.85, tol=1.E-6, normalization="auto") >> pg.Ordinals()
ranks = ranker(graph, {v: 1 for v in seeds})
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
early_stop_ranker = pg.PageRank(alpha=0.85, convergence=convergence) >> pg.Ordinals()
ordinals = early_stop_ranker(graph, {v: 1 for v in seeds})
print(early_stop_ranker.convergence)
# 2 iterations (0.0005241000035312027 sec)
print(ordinals["B"], ordinals["D"], ordinals["E"])
# 3.0 5.0 4.0
```

This is close to the previous results at a fraction of the time!
For large graphs,
most ordinals would be near the ideal ones. Note that convergence time 
does not take into account the time needed to preprocess graphs.
