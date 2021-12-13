![pygrank](documentation/pygrank.png)

Fast recommendation algorithms for large graphs based on link analysis.
<br>
<br>

**License:** Apache Software License
<br>**Author:** Emmanouil (Manios) Krasanakis
<br>**Dependencies:** `networkx, numpy, scipy, sklearn, wget` (required) `tensorflow`, `torch` (optional)

![build](https://github.com/MKLab-ITI/pygrank/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/MKLab-ITI/pygrank/branch/master/graph/badge.svg?token=OWGHRGDYOP)](https://codecov.io/gh/maniospas/pygrank)
[![Downloads](https://static.pepy.tech/personalized-badge/pygrank?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads)](https://pepy.tech/project/pygrank)

# :tent: Roadmap for 0.2.X
The following roadmap overviews short-term development goals and will be updated appropriately.

:heavy_check_mark: Reach a stable architecture with comprehensive development management (achieved as of 0.2.3, no longer backwards compatible with 0.1.17, most important change `to_scipy` >> `preprocessor`.) <br>
:heavy_check_mark: Graph neural network support with dropout, renormalization and tensorflow backend (achieved as of 0.2.3)<br>
:heavy_check_mark: Pytorch backend (achieved as of 0.2.4)<br>
:x: Pytorch gnns<br>
:x: 100% code coverage<br>
:x: 100% documentation completeness<br>
:x: Automatic download for all related publication datasets ([downloadable datasets](documentation/datasets.md))<br>
:heavy_check_mark: Automated citation discovery for algorithms (achieved as of 0.2.5 with `NodeRanking.cite()`)<br>
:heavy_check_mark: Enable Arnoldi and Lanczos optimizations in non-numpy backends (achieved as of 0.2.5)<br>
:x: Transparent handling of float and double precisions (as of 0.2.5 numpy is on float64 but pytorch and tensorflow in float32)

# :hammer_and_wrench: Installation
`pygrank` is meant to work with Python 3.9 or later. The latest version can be installed with pip per:
```
pip install --upgrade pygrank
```

To automatically use the machine learning backends (e.g. to integrate the package
in machine learning projects) *tensorflow* and *pytorch*,
 manually change the automatically created
configuration file whose path is displayed in the error console.
If you want others to run your code that depends on `pygrank`
with specific backends, add the following recipe at your code's
entry point to override other configurations:

```python
import pygrank as pg
pg.load_backend(`pytorch`)
```


# :zap: Quickstart
As a quick start, let us construct a networkx graph `G` and a set of nodes `seeds`.

```python
import networkx as nx
graph = nx.Graph()
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
 We start by importing the library:

```python
import pygrank as pg
```

For instructional purposes,
we experiment with (personalized) *PageRank*. This and more filters can be found in the module
`pygrank.algorithms.filters`, but for ease-of-use can
be accessed from the top-level import.
We also set the default values of some parameters: the graph diffusion
rate *alpha* required by the filter, a numerical tolerance *tol* at the
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
# 0.25865456609095644 0.12484722044728883 0.17079023174039495
```

We alter this outcome so that it outputs node order, 
where higher node scores are assigned lower order,
by wrapping a postprocessor around the base algorithm. There are various
postprocessors, including ones to make scores fairness-aware. Again,
postprocessors can be found in `pygrank.algorithms.postprocess`,
but can be accessed from the top-level package import.

```python
ordinals = pg.Ordinals(ranker).rank(graph, {v: 1 for v in seeds})
print(ordinals["B"], ordinals["D"], ordinals["E"])
# 1.0 5.0 4.0
```

How much time did it take for the base ranker to converge?
(Depends on backend and device characteristics.)

```python
print(ranker.convergence)
# 19 iterations (0.0020614000000023225 sec)
```

Since only the node order is important,
we can use a different way to specify convergence:

```python
convergence = pg.RankOrderConvergenceManager(pagerank_alpha=0.85, confidence=0.98) 
early_stop_ranker = pg.PageRank(alpha=0.85, convergence=convergence)
ordinals = pg.Ordinals(early_stop_ranker).rank(graph, {v: 1 for v in seeds})
print(early_stop_ranker.convergence)
# 2 iterations (0.0006666000000450367 sec)
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
# 1.0 5.0 4.0
```

This yields the same node ordinals, which means that tuning constructed
a graph filter similar to `PageRank`.
Tuning may be worse than highly specialized algorithms in some settings, but often
finds near-best base algorithms.

To obtain a recommendation about how to cite complex
algorithms, an automated description can be extracted by the source code per the following
command, where bibtex entries corresponding to the citations can be found 
[list of citations](documentation/citations.md).:

```python
print(tuned_ranker.cite())
# graph filter \cite{ortega2018graph} with dictionary-based hashing \cite{krasanakis2021pygrank} and parameters tuned \cite{krasanakis2021pygrank} to optimize AUC while withholding 0.200 of nodes for validation
```


# :brain: Overview
Analyzing graph edges (links) between graph nodes can help rank/score
them based on proximity to structural or attribute-based communities of known example members.
With the introduction of graph signal processing and
[decoupled graph neural networks](https://dl.acm.org/doi/abs/10.1145/3442381.3449927) 
the importance of node ranking has drastically 
increased, as its ability to perform inductive learning by quickly
spreading node information through edges has been theoretically and experimentally
corroborated. For example, it can be used to make predictions based on few known
node attributes or the outputs by feature-based machine learning models.

`pygrank` is a collection of node ranking algorithms and practices that 
support real-world conditions, such as large graphs and heterogeneous
preprocessing and postprocessing requiremenets. Thus, it provides
ready-to-use tools that simplify deployment of theoretical advancements
and testing of new algorithms.

Some of the library's advantages are:
1. **Compatibility** with [networkx](https://github.com/networkx/networkx), [tensorflow](https://www.tensorflow.org/) and [pytorch](https://pytorch.org/).
2. **Datacentric** interfaces that do not require transformations to identifiers.
3. **Large** graph support with sparse representations and fast algorithms.
4. **Seamless** pipelines, from graph preprocessing up to benchmarking and evaluation.
5. **Modular** combination of components.


# :link: Material
[Tutorials & Documentation](documentation/documentation.md)

**Quick links**<br>
[Measures](documentation/measures.md)<br>
[Graph Filters](documentation/graph_filters.md)<br>
[Postprocessors](documentation/postprocessors.md)<br>
[Tuners](documentation/tuners.md)<br>
[Downloadable Datasets](documentation/datasets.md)<br>

# :fire: Features
* Graph filters
* Community detection
* Graph normalization
* Convergence criteria
* Postprocessing (e.g. fairness awareness)
* Evaluation measures
* Benchmarks
* Autotuning

# :thumbsup: Contributing
Feel free to contribute in any way, for example through the [issue tracker](https://github.com/MKLab-ITI/pygrank/issues) or by participating in [discussions]().
Please check out the [contribution guidelines](CONTRIBUTING.md) to bring modifications to the code base.
If so, make sure to **follow the pull checklist** described in the guidelines.
 
# :notebook: Citation
If `pygrank` has been useful in your research and you would like to cite it in a scientific publication, please refer to the following paper:
```
@misc{krasanakis2021pygrank,
      title={pygrank: A Python Package for Graph Node Ranking}, 
      author={Emmanouil Krasanakis, Symeon Papadopoulos, Ioannis Kompatsiaris, Andreas Symeonidis},
      year={2021},
      eprint={2110.09274},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
To publish research that makes use provided methods,
please cite all [relevant publications](documentation/citations.md).
