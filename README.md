# pygrank
Fast recommendation algorithms for large graphs based on link analysis.
This library implements popular graph filters to perform node
recommendation, postprocessing approaches to improve recommendation quality 
and make its outcome fairer, as well as supervised and unsupervised
measures of recommendation quality.

## Table of Contents
* [Table of Contents](#table-of-contents)
* [Installation](#installation)
* [Usage](#usage)
    + [Ranking Algorithms](#ranking-algorithms)
    + [Adjacency Matrix Normalization](#adjacency-matrix-normalization)
    + [Augmenting Node Ranks](#augmenting-node-ranks)
    + [Convergence Criteria](#convergence-criteria)
    + [Rank Quality Evaluation](#rank-quality-evaluation)
* [References](#references)
    + [Glossary](#glossary)
    + [Method References](#method-references)
    + [Published](#publications)
    + [Under Review](#under-review)
    + [Related](#related)

## Installation
```
pip install pygrank
```

## Usage

### Ranking Algorithms
Ranking algorithms assume that there exists a networkx graph `G`
and a non-negative personalization dictionary of node importances
(missing nodes are considered to have 0 importance). For example,
if we consider a list `edges` of edge tuples and a list `seedS` of
nodes towards which to consider structural importance, the graph
and personalization can be constructed as:
```python
import networkx as nx
edges, seeds = ...
G = nx.Graph()
for u, v in edges:
    G.add_edge(u, v)
personalization = {v: 1 for v in seeds}
```

Given the above way of creating a graph and a personalization
dictionary (which is the programming artifact equivalent to what
the literature referred to as a graph signal or personalization vector), 
one can instantiate a ranking algorithm class 
and call its `rank(G, personalization)` method to capture
node ranks in a given graph with the given personalization. This
method exists for all ranking algorithms and potential wrappers
that aim to augment ranks in some way 
(see [Improving Ranking Outcome](#improving-ranking-outcome)). 
Algorithmic parameters, such as convergence critertia and
the type of adjacency matrix normalization, are passed to the
constructor of the ranking algorithm. As an example, running a
personalized PageRank algorithm with diffusion rate `alpha=0.85`
with absolute rank error tolerance `tol=1.E-6` (its default parameters)
and printing its node ranks can be done as follows:

```python
from pygrank.algorithms.pagerank import PageRank
G, personalization = ...
ranker = PageRank(alpha=0.85, tol=1.E-6)
ranks = ranker.rank(G, personalization)
print('Convergence report:', str(ranker.convergence))
for v, rank in ranks.items():
    print('The rank of node', v, 'is', rank)
```


:bulb: For general-purpose usage, we recommend trying
`PageRank(0.85)`, `PageRank(0.95)` or `HeatKernel(3)`,
all of which capture some commonly found types of 
rank propagation. If these are used on large graphs (with
thousands or milions of nodes), we recommend passing a
stricter tolerance parameter  `tol1.E-9` to constructors
to make sure that the personalization is propagated to most nodes.


:bulb: For even faster running speeds that avoid conversion of
dictionaries to numpy arrays and conversely,
pass the argument ``as_dict=False`` to  the``rank(...)`` 
method of graph ranking algorithms to make them output a numpy
array (with elements in the same order as the order of nodes 
in the networkx graph, e.g. the order of traversing ``for u in G``).
Similarly, numpy arrays can also be passed to that method instead
of the personalization dictionary to also avoid these conversions.
**This optimization is not supported by some post-processing schemes.**


### Adjacency Matrix Normalization
Node ranking algorithms all use the same default scheme
that performs symmetric (i.e. Lalplacian-like) normalization 
for undirected graphs and column-wise normalization that
follows a true probabilistic formulation of transition probabilities
for directed graphs, such as `DiGraph` instances. The type of
normalization can be manually edited by passing a `normalization`
argument to constructors of ranking algorithms. This parameter can 
assume values of "auto" for the above-described default behavior, 
"col" for column-wise normalization, "symmetric" for symmetric 
normalization and "none" for avoiding any normalization, 
for example because edge weights already hold the normalization
(e.g. this is used to rank graphs after FairWalk is used to
preprocess edge weights).

In all cases, adjacency matrix normalization involves the
computationally intensive operation of converting the graph 
into a scipy sparse matrix each time  the `rank(G, personalization)`
method of ranking algorithms is called. The *pygrank* library
provides a way to avoid recomputing the normalization
during large-scale experiments by the same algorithm for 
the same graphs by passing an argument `assume_immutability=True`
to the algorithms's constructor, which indicates that
the the graph does not change between runs of the algorithm
and hence computes the normalization only once for each given
graph, a process known as hashing.

:warning: Hashing only uses the Python object's hash method, 
so a different instance of the same graph will recompute the 
normalization if it points at a different memory location.

:warning: Do not alter graph objects after passing them to
`rank(...)` methods of algorithms with
`assume_immutability=True` for the first time. If altering the
graph is necessary midway through your code, create a copy
instance with one of *networkx*'s in-built methods and
edit that one.

For example, hashing the outcome of graph normalization to
speed up multiple calls to the same graph can be achieved
as per the following code:
```python
from pygrank.algorithms.pagerank import PageRank
G, personalization1, personalization2 = ...
algorithm = PageRank(alpha=0.85, normalization="col", assume_immutability=True)
ranks = algorithm.rank(G, personalization1)
ranks = algorithm.rank(G, personalization2) # does not re-compute the normalization
```

Sometimes, many different algorithms are applied on the
same graph. In this case, to prevent each one
from recomputing the hashing already calculated by others,
they can be made to share the same normalization method. This 
can be done by using a shared instance of the 
normalization preprocessing class `preprocessor`, 
which can be passed as the `to_scipy` argument of ranking algorithm
constructors. In this case, the `normalization` and `assume_immutability`
arguments should be passed to the preprocessor and will be ignored by the
constructors (what would otherwise happen is that the constructors
would create a prerpocessor with these arguments).

:bulb: Basically, when the default value `to_scipy=None`
is given, ranking algorithms create a new preprocessing instance
with the `normalization` and `assume_immutability` values passed
to their constructor. These two arguments are completely ignored
if a preprocessor instance is passed to the ranking algorithm.

For example, using the outcome of graph normalization 
to speed up multiple rank calls to the same graph by
different ranking algorithms can be done as:
```python
from pygrank.algorithms.pagerank import PageRank, HeatKernel
from pygrank.algorithms.utils import preprocessor
G, personalization1, personalization2 = ...
pre = preprocessor(normalization="col", assume_immutability=True)
ranker1 = PageRank(alpha=0.85, to_scipy=pre)
ranker2 = HeatKernel(alpha=0.85, to_scipy=pre)
ranks1 = ranker1.rank(G, personalization1)
ranks2 = ranker2.rank(G, personalization2) # does not re-compute the normalization
```

### Augmenting Node Ranks
It is often desirable to postprocess the outcome of node ranking
algorithms. This can be some simple normalization or involve more 
complex procedures, such as making ranks protect a group of sensitive
nodes (hence making them fairness-aware).

The simpler postprocessing mechanisms only aim to transform
outputted ranks, for example by normalizing them or outputting
their ordinality. For example, the following code wraps the base algorithm
with a preprocessor that assigns rank 1 to the highest scored node,
2 to the second highest, etc:

```python
from pygrank.algorithms.postprocess import Ordinals

G, personalization = ...
base_algorithm = ... # e.g. PageRank

algorithm = Ordinals(base_algorithm)
ordinals = algorithm.rank(G, personalization)
```

For ease of use, this type of postprocessing also provides a transformation
method that can be directly used to transform ranks without wrapping the
base algotithm. For example, the following code performs the same operation
as the previous one:

```python
from pygrank.algorithms.postprocess import Ordinals

G, personalization = ...
base_algorithm = ... # e.g. PageRank

ordinals = Ordinals().transform(base_algorithm.rank(G, personalization))
```


A second type of postprocessing uses computed ranks to change (e.g. edit)
the personalization before re-running the ranking algorithms, sometimes 
multiple times. For example, the following seed oversampling scheme we first
introduced in \[krasanakis2019boosted\] uses a ``base_algorithm`` to rank nodes
and then sets seeds to one for nodes with higher ranks than any of the original seeds
and then reruns that algorithm with updated seeds:

```python
from pygrank.algorithms.oversampling import SeedOversampling

G, personalization = ...
base_algorithm = ... # e.g. PageRank

algorithm = SeedOversampling(base_algorithm)
ranks = algorithm.rank(G, personalization)
```

### Convergence Criteria
Most base ranking algorithm constructors allow ``convergence`` argument that
indicates an object to help determing their convergence criteria, such as type of
error and tolerance for numerical convergence. If no such argument is passed
to the constructor, a ``pygrank.algorithms.utils.ConvergenceManager`` object
is automatically instantiated by borrowing whichever extra arguments it can
from those passed to the constructors. Most frequently used is the ``tol``
argument to indicate the numerical tolerance level required for convergence.

Sometimes, it suffices to reach a robust node rank order  instead of precise 
values. To cover such cases we have implemented a different convergence criterion
``pygrank.algorithms.utils.RankOrderConvergenceManager`` that stops 
at a robust node order \[krasanakis2020stopping\].


:warning: This criterion is specifically intended to be used with PageRank 
as the base ranking algorithm and needs to know that algorithm's diffusion
rate ``alpha``, which is passed as its first argument.

```python
from pygrank.algorithms.pagerank import PageRank
from pygrank.algorithms.utils import RankOrderConvergenceManager
from pygrank.algorithms.postprocess import Ordinals

G, personalization = ...
alpha = 0.85
ordered_ranker = PageRank(alpha=alpha, convergence=RankOrderConvergenceManager(alpha))
ordered_ranker = Ordinals(ordered_ranker)
ordered_ranks = ordered_ranker.rank(G, personalization)
```

:bulb: Since the node order is more important than the specific rank values,
a post-processing step has been added throught the wrapping expression
``ordered_ranker = Ordinals(ordered_ranker)`` to output rank order. 


### Rank Quality Evaluation
There are two types of node rank evaluations; supervised and unsupervised.
The evaluation process assumes that nodes form structural or ground truth
communities out of which a few seed nodes are known. If other nodes are also
known, they can be used for supervised evaluation, otherwise unsupervised 
metrics need to be selected.

:bulb: Supervised metrics can also evaluate numpy arrays obtained from the
``as_dict=False`` ranking argument but support for this feature is still
limited for ther metrics. More extensive documentation of this feature
will be provided in the future.

###### How to evaluate ranks with an unsupervised metric
```python
from pygrank.algorithms.postprocess import Normalize
from pygrank.metrics.unsupervised import Conductance

G, ranks = ... # calculate as per the first example
normalized_ranks = Normalize().transform(ranks)

metric = Conductance(G)
print(metric.evaluate(normalized_ranks))
```

###### How to evaluate ranks with a supervised metric
```python
from pygrank.metrics.supervised import AUC
import pygrank.metrics.utils

G, seeds, algorithm = ... # as per the first example
seeds, ground_truth = pygrank.metrics.utils.split_groups(seeds, fraction_of_training=0.5)

pygrank.metrics.utils.remove_group_edges_from_graph(G, ground_truth)
ranks = algorithm.rank(G, {v: 1 for v in seeds})

metric = AUC({v: 1 for v in ground_truth})
print(metric.evaluate(ranks))
```

###### How to evaluate multiple ranks
```python
import networkx as nx
from pygrank.algorithms.pagerank import PageRank as Ranker
from pygrank.algorithms.postprocess import Normalize as Normalizer
from pygrank.algorithms.oversampling import BoostedSeedOversampling as Oversampler
from pygrank.metrics.unsupervised import Conductance
from pygrank.metrics.supervised import AUC
from pygrank.metrics.multigroup import MultiUnsupervised, MultiSupervised, LinkAUC
import pygrank.metrics.utils

# Construct data
G = nx.Graph()
groups = {}
groups["group1"] = list()
... 

# Split to training and test data
training_groups, test_groups = pygrank.metrics.utils.split_groups(groups)
pygrank.metrics.utils.remove_group_edges_from_graph(G, test_groups)

# Calculate ranks and put them in a map
algorithm = Normalizer(Oversampler(Ranker(alpha=0.99)))
ranks = {group_id: algorithm.rank(G, {v: 1 for v in group}) 
        for group_id, group in training_groups.items()}
        

# Evaluation with Conductance
conductance = MultiUnsupervised(Conductance, G)
print(conductance.evaluate(ranks))

# Evaluation with LinkAUC
link_AUC = LinkAUC(G, pygrank.metrics.utils.to_nodes(test_groups))
print(link_AUC.evaluate(ranks))

# Evaluation with AUC
auc = MultiSupervised(AUC, pygrank.metrics.utils.to_seeds(test_groups))
print(auc.evaluate(ranks))
        
```


## References
### Glossary
- *Seeds.* Example nodes that are known to belong to a community.
- *Ranks.* Scores (not ordinalities) assigned to nodes. They typically assume
values in the range \[0,1\].
- *Personalization.* A hashmap between seeds and scores to be passed to graph ranking algorithms. This is also known as a personalization vector
or graph signal priors. 
- *Node ranking algorithm.* An algorithm that starts with a graph and personalization and outputs a hashmap of node scores.

### Method References

Instantiation or Usage | Method Name | Citation
--- | --- | --- 
`pygrank.algorithms.oversampling.SeedOversampling(ranker)` | SeedO | krasanakis2019boosted
`pygrank.algorithms.oversampling.BoostedSeedOversampling(ranker)` | SeedBO | krasanakis2019boosted
`pygrank.algorithms.pagerank.PageRank(converge_to_eigenvectors=True)` | VenueRank | krasanakis2018venuerank
`G = pygrank.postprocess.fairness.to_fairwalk(G, sensitive)` | FairWalk |rahman2019fairwalk
`pygrank.algorithms.postprocess.fairness.FairPostprocessor(ranker,'O')` | LFPRO | tsioutsiouliklis2020fairness
`pygrank.algorithms.postprocess.fairness.FairPersonalizer(ranker, error_type="mabs", max_residual=0)` | FP | krasanakis2020fairconstr
`pygrank.algorithms.postprocess.fairness.FairPersonalizer(ranker, 0.8, 10, error_type="mabs", max_residual=0)` | CFP | krasanakis2020fairconstr
`pygrank.algorithms.postprocess.fairness.FairPersonalizer(ranker)` | FairEdit | krasanakis2020prioredit
`pygrank.algorithms.postprocess.fairness.FairPersonalizer(ranker, 0.8, 10)` | FairEditC | krasanakis2020prioredit
`pygrank.metrics.multigroup.LinkAUC(G, hops=1)` | LinkAUC | krasanakis2019linkauc
`pygrank.metrics.multigroup.LinkAUC(G, hops=2)` | HopAUC | krasanakis2020unsupervised
`pygrank.algorithms.utils.RankOrderConvergenceManager(alpha, confidence=0.99, criterion="fraction_of_walks")` | | krasanakis2020stopping
`pygrank.algorithms.utils.RankOrderConvergenceManager(alpha)` | | krasanakis2020stopping

### Publications
The publications that have led to the development of various aspects of
this library are presented in reverse chronological order.
```
@article{krasanakis2020unsupervised,
  title={Unsupervised evaluation of multiple node ranks by reconstructing local structures},
  author={Krasanakis, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Yiannis},
  journal={Applied Network Science},
  volume={5},
  number={1},
  pages={1--32},
  year={2020},
  publisher={Springer}
}
```
```
@article{krasanakis2019boosted,
  title={Boosted seed oversampling for local community ranking},
  author={Krasanakis, Emmanouil and Schinas, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Yiannis and Symeonidis, Andreas},
  journal={Information Processing \& Management},
  pages={102053},
  year={2019},
  publisher={Elsevier}
}
```
```
@inproceedings{krasanakis2020stopping,
  title={Stopping Personalized PageRank without an Error Tolerance Parameter},
  author={Krasanakis, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Ioannis},
  year={2020},
  booktitle={ASONAM},
}
```
```
@inproceedings{krasanakis2020fairconstr,
  title={Applying Fairness Constraints on Graph Node Ranks under Personalization Bias},
  author={Krasanakis, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Ioannis},
  year={2020},
  booktitle={Complex Networks},
}
```
```
@inproceedings{krasanakis2019linkauc,
  title={LinkAUC: Unsupervised Evaluation of Multiple Network Node Ranks Using Link Prediction},
  author={Krasanakis, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Yiannis},
  booktitle={International Conference on Complex Networks and Their Applications},
  pages={3--14},
  year={2019},
  organization={Springer}
}
```
```
@inproceedings{krasanakis2018venuerank,
  title={VenueRank: Identifying Venues that Contribute to Artist Popularity},
  author={Krasanakis, Emmanouil and Schinas, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Yiannis and Mitkas, Pericles A},
  booktitle={ISMIR},
  pages={702--708},
  year={2018}
}
```

### Under Review
```
@inproceedings{krasanakis2020prioredit,
  title={Prior Signal Editing for Graph Filter Posterior Fairness Constraints},
  author={Krasanakis, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Ioannis},
  booktitle={under review},
}
```

### Related
Here, we list additional publications whose methods are fully or partially implemented in this library.
```
@article{tsioutsiouliklis2020fairness,
  title={Fairness-Aware Link Analysis},
  author={Tsioutsiouliklis, Sotiris and Pitoura, Evaggelia and Tsaparas, Panayiotis and Kleftakis, Ilias and Mamoulis, Nikos},
  journal={arXiv preprint arXiv:2005.14431},
  year={2020}
}
```
```
@inproceedings{rahman2019fairwalk,
  title={Fairwalk: Towards Fair Graph Embedding.},
  author={Rahman, Tahleen A and Surma, Bartlomiej and Backes, Michael and Zhang, Yang},
  booktitle={IJCAI},
  pages={3289--3295},
  year={2019}
}
```