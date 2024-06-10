# Quickstart

## 1. Install and import
Install the library using `pip install pygrank` and import it. Construct a node ranking algorithm from a graph filter by incrementally applying postprocessors using >>. There are many components and parameters available. Use [autotuning](autotuning.md) to find good configurations.

```python
import pygrank as pg

hk5 = pg.HeatKernel(t=5, normalization="symmetric", renormalize=True)  # a graph filter
hk5_advanced = hk5 >> pg.SeedOversampling() >> pg.Sweep() >> pg.Normalize("max") 
```

## 2. Load a graph and community
Automatically load a graph and a community of nodes with a shared attribute. You can also create a custom `networkx` graph. Run the algorithm to get a graph signal that maps nodes to scores indicating structural proximity to community members.

```python
_, graph, community = next(pg.load_datasets_one_community(["eucore"]))
personalization = {node: 1.0 for node in community}  # binary or stochastic membership, missing scores are zero

scores = hk5_advanced(graph, personalization)  # returns a dict-like pg.GraphSignal
print(scores)  # {'0': 0.3154503251398683, '1': 0.26661671252340463, '2': 0.03700150026429704, ... }
```

## 3. Evaluate
Evaluate the scores using a stochastic generalization of the unsupervised conductance measure.

```python
measure = pg.Conductance()  # an evaluation measure
pg.benchmark_print_line("My conductance", measure(scores))  # pretty
print("Cite this algorithm as:", hk5_advanced.cite())
```
