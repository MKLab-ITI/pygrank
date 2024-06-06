# Quickstart

1.&nbsp;Install the library with `pip install pygrank`, import it,
and construct a node ranking algorithm
(incrementally apply postprocessors with `>>`). 
There are many components and parameters to find 
good configurations; [autotuning](../advanced/autotuning.md) may be helpful.

```python
import pygrank as pg

hk5 = pg.HeatKernel(t=5, normalization="symmetric", renormalize=True)  # a graph filter
hk5_advanced = hk5 >> pg.SeedOversampling() >> pg.Sweep() >> pg.Normalize("max") 
```

2.&nbsp;Automatically load a graph and a community of nodes with some shared attribute. 
You can also use a `networkx` graph. 
Then run the algorithm to get a graph signal that maps nodes to scores, where scores indicate
structural proximity to community members.

```python
_, graph, community = next(pg.load_datasets_one_community(["eucore"]))
personalization = {node: 1.0 for node in community}  # binary or stochastic membership, missing scores are zero

scores = hk5_advanced(graph, personalization)  # returns a dict-like pg.GraphSignal
print(scores)  # {'0': 0.3154503251398683, '1': 0.26661671252340463, '2': 0.03700150026429704, ... }
```

3.&nbsp;Evaluate scores; here we use a stochastic generalization of the unsupervised Conductance measure (that
can parse scores).

```python
measure = pg.Conductance()  # an evaluation measure
pg.benchmark_print_line("My conductance", measure(scores))  # pretty
print("Cite this algorithm as:", hk5_advanced.cite())
```
