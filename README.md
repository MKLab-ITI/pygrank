![pygrank](documentation/pygrank.png)

Fast recommendation algorithms for large graphs based on link analysis.
<br>
<br>

**License:** Apache Software License
<br>**Author:** Emmanouil (Manios) Krasanakis
<br>**Dependencies:** `networkx, numpy, scipy, sklearn, wget`
<br>**Backends:** `tensorflow`, `torch`, `torch_sparse`, `matvec` (optional)
<br><small>*Externally install backends before using them.*</small>

![build](https://github.com/MKLab-ITI/pygrank/actions/workflows/tests.yml/badge.svg)
[![codecov](https://codecov.io/gh/MKLab-ITI/pygrank/branch/master/graph/badge.svg?token=RYZOT4UY8Q)](https://codecov.io/gh/MKLab-ITI/pygrank)
[![Downloads](https://static.pepy.tech/personalized-badge/pygrank?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads)](https://pepy.tech/project/pygrank)

# :rocket: New features (after 0.2.7)
* Algorithm definition [chains](documentation/functional.md)
* More [downloadable datasets](documentation/datasets.md)
* New graph management for minimal memory consumption
* Using preprocessor outcomes as graphs

# :hammer_and_wrench: Installation
`pygrank` is meant to work with Python 3.9 or later. The latest version can be installed with pip per:
```
pip install --upgrade pygrank
```

To run the library on backpropagateable machine learning backends, 
namely *tensorflow* or *pytorch*, either change the automatically created
configuration file or run parts of your code within the following
[context manager](https://book.pythontips.com/en/latest/context_managers.html)
to override other configurations.
Replace *tensorflow* with other desired backend names:

```python
import pygrank as pg
with pg.Backend("tensorflow"):
    ... # run your pygrank code here
```

If you do nothing, everything runs on top of `numpy` (currently, this
is faster for forward passes).
The library's algorithms can be defined before contexts and only
be called inside them. You can also use the simpler
`pg.load_backend("tensorflow")` to switch to a specific backend
if you want to avoid contexts.

# :zap: Quickstart
Before looking at the library's details, we show a fully functional
pipeline that can rank the importance of a node in relation to 
a list of "seed" nodes within a graph's structure:

```python
import pygrank as pg
graph, seeds, node = ...

pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
algorithm = pg.PageRank(alpha=0.85)+pre >> pg.Sweep() >> pg.Ordinals()
ranks = algorithm(graph, seeds)
print(ranks[node])
print(algorithm.cite())
```

The graph can be created with `networkx` or, for faster computations,
with the library itself. Nodes can hold any 
kind of object or data type. You don't need to bother with
conversion to integer identifiers - the library does this
internally and very fastly.

The above snippet first defines a `preprocessor`, 
which controls how graph adjacency matrices will be normalized 
by related algorithms. In this case, a symmetric normalization
takes place (which is ideal for undirected graphs) and we also
assume graph immutability to hash the preprocessor's outcome
so that it is not recomputed every time we experiment with the
same graphs.

The snippet makes use of the library's 
[chain operators](documentation/functional.md)
to wrap node ranking algorithms by various kinds of postprocessors
with the `>>` operator
(you can also put algorithms into each other's constructors
if you are not a fan of functional programming).
The chain starts from a pagerank graph filter with diffusion parameter
0.85. Other types of filters and even automatically tuned ones
can be run.

Then, the algorithm is run as a callable,
producing a map between nodes and values 
(in graph signal processing, such maps are called graph signals)
and we print the value of a particular node. Graph signals can
also be created and directly parsed by algorithms, for example per:
```
signal = pg.to_signal(graph, {v: 1. for v in seeds})
ranks = algorithm(signal)
```

Finally, the snippet prints a recommended citation for the algorithm.

### More examples

[Simple example](documentation/showcase.md) <br>
[Big data FAQ](documentation/tips.md) <br>
[Downstream tasks](https://github.com/maniospas/pygrank-downstream) <br>


# :brain: Overview
Analyzing graph edges (links) between graph nodes can help rank/score
them based on proximity to structural or attribute-based communities of known example members.
With the introduction of graph signal processing and
[decoupled graph neural networks](https://dl.acm.org/doi/abs/10.1145/3442381.3449927) 
the importance of node ranking has drastically 
increased, as its ability to perform inductive learning by quickly
spreading node information through edges has been theoretically and experimentally
corroborated. For example, it can be used to make predictions based on few known
node attributes or the outputs of feature-based machine learning models.

`pygrank` is a collection of node ranking algorithms and practices that 
support real-world conditions, such as large graphs and heterogeneous
preprocessing and postprocessing requirements. Thus, it provides
ready-to-use tools that simplify deployment of theoretical advancements
and testing of new algorithms.

Some of the library's advantages are:
1. **Compatibility** with [networkx](https://github.com/networkx/networkx), [tensorflow](https://www.tensorflow.org/) and [pytorch](https://pytorch.org/).
2. **Datacentric** interfaces that do not require transformations to identifiers.
3. **Large** graph support with sparse data structures and fast scalable algorithms.
4. **Seamless** pipelines, from graph preprocessing up to benchmarking and evaluation.
5. **Modular** components to be combined and a functional chain interface for complex combinations.


# :link: Material
[Tutorials & Documentation](documentation/documentation.md) <br>
[Functional Interface](documentation/functional.md)

**Quick links**<br>
[Measures](documentation/measures.md) <br>
[Graph Filters](documentation/graph_filters.md) <br>
[Postprocessors](documentation/postprocessors.md) <br>
[Tuners](documentation/tuners.md) <br>
[Downloadable Datasets](documentation/datasets.md) <br>

**Backend resources**<br>
[numpy](https://numpy.org/) (default, no additional installation) <br>
[tensorflow](https://www.tensorflow.org/install) <br>
[pytorch](https://pytorch.org/get-started/locally) <br>
[torch_sparse](https://github.com/rusty1s/pytorch_sparse) <br>
[matvec](https://github.com/maniospas/matvec)

# :fire: Features
* Graph filters
* Community detection
* Overlapping community detection
* Link prediction
* Graph normalization
* Convergence criteria
* Postprocessing (e.g. fairness awareness)
* Evaluation measures
* Benchmarks
* Autotuning
* Graph Neural Network (GNN) support

# :thumbsup: Contributing
Feel free to contribute in any way, for example through the [issue tracker](https://github.com/MKLab-ITI/pygrank/issues) or by participating in [discussions]().
Please check out the [contribution guidelines](CONTRIBUTING.md) to bring modifications to the code base.
If so, make sure to **follow the pull checklist** described in the guidelines.
 
# :notebook: Citation
If `pygrank` has been useful in your research and you would like to cite it in a scientific publication, please refer to the following paper:
```
@article{emmanouil_krasanakis_2022_7229677,
  author       = {Emmanouil Krasanakis, Symeon Papadopoulos, Ioannis Kompatsiaris, Andreas Symeonidis},
  title        = {pygrank: A Python Package for Graph Node Ranking},
  journal      = {SoftwareX},
  year         = 2022,
  month        = oct,
  doi          = {10.1016/j.softx.2022.101227},
  url          = {https://doi.org/10.1016/j.softx.2022.101227}
}
```
To publish research that makes use of provided methods,
please cite all [relevant publications](documentation/citations.md).
