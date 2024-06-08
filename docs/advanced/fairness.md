# Fairness

Fairness-aware graph filters and
postprocessors are provided. Below
we provide two example node ranking algorithms
that make personalized PageRank fairer by
supplying it with a postprocessor or straight
up using a variation.

```python
import pygrank as pg

ppro = pg.PageRank(alpha=0.9, max_iters=1000, tol=1.E-9) >> pg.AdHocFairness("O")
lfpr = pg.LFPR(alpha=0.9, max_iters=1000, tol=1.E-9)
```

When calling fairness-aware methods, add a *sensitive* keyword
argument that holds information that together with a graph
can construct a graph signal.

Several measures also provide fairness assessment.

```python
_, graph, groups = next(pg.load_datasets_multiple_communities(['citeseer'], max_group_number=2, directed=False))
measure = pg.pRule(groups[1])

pg.benchmark_print_line("ppro", ppro(graph, groups[0], sensitive=groups[1]) >> measure)
pg.benchmark_print_line("lfpr", lfpr(graph, groups[0], sensitive=groups[1]) >> measure)
```

!!! info
    For multidimensional fairness evaluation of node
    ranking algorithms, prefer using the
    `fairbench` library.