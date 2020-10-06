# pygrank
Recommendation algorithms for large graphs.

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
dictionary (which is equivalent to the personalization vector
referred to the literature), one can instantiate a ranking algorithm
class and call its `rank(G, personalization)` method to capture
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
stricter tolerance parameter  `tol1.E-9` to these constructors
to make sure that the poersonalization is propagated to most nodes.


### Adjacency Matrix Normalization
Node ranking algorithms all use the same default scheme
that performs symmetric (i.e. Lalplacian-like) normalization 
for undirected graphs and column-wise normalization that
follows a true probabilistic formulation of transition probabilities
for directed graphs, such as `DiGraph` instances. The type of
normalization can be manually edited by passing a `normalization`
argument to the constructor of ranking algorithms, which can assume
values of "auto" for the default behavior, "col" for column-wise
normalization, "symmetric" for symmetric normalization and "none"
for avoiding any normalization, for example because it was performed
and set to edge weights.

In all cases, ajacency matrix normalization involves the
computationally intensive operations of converting the graph 
into a scipy sparse matrix each time  the `rank(G, personalization)`
method of ranking algorithms is  called. The *pygrank* library
provides a way to avoid recomputing the normalization
during large-scale experiments by the same algorithm for 
the same graphs by passing an argument `assume_immutability=True`
to the algorithms's constructor, which indicates that
the the graph does not change between runs of the algorithm
and hence computes the normalization only once for each given
graph, a process known as hashing.

:warning: Do not alter graph objects after passing them to
a `rank(...)` method of algorithms with
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
same graph. In this case, to prevent each algorithm
from recomputing the hashing already calculated by others,
they can be made to share the same normalization method. This 
can be done by using a shared instance of the (hashed) 
normalization preprocessing, which can be passed as the
`to_scipy` argument to their constructor instead of using
the previous. 

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
Several approaches aim to postprocess the outcome of node ranking
algorithms. This postprocessing may vary from simple normalization
to more complex processes.

The first type of

```python
from pygrank.algorithms.postprocess import Ordinals

G, personalization = ...
base_algorithm = ... # e.g. PageRank

algorithm = Ordinals(base_algorithm)
ordinals = algorithm.rank(G, personalization)
```


An additional type of methods
```python
from pygrank.algorithms.oversampling import SeedOversampling

G, personalization = ...
base_algorithm = ... # e.g. PageRank

algorithm = SeedOversampling(base_algorithm)
ranks = algorithm.rank(G, personalization)
```

### Convergence Criteria
Run a PageRank algorithm and make it converge to a robust node order
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
a post-processing step to map nodes to that order can be added to the algorithm as:



### Rank Quality Evaluation

###### How to evaluate with an unsupervised metric
```python
from pygrank.algorithms.postprocess import Normalize
from pygrank.metrics.unsupervised import Conductance

G, ranks = ... # calculate as per the first example
normalized_ranks = Normalize().rank(ranks)

metric = Conductance(G)
print(metric.evaluate(normalized_ranks))
```

###### How to evaluate with a supervised metric
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
### Method References

Instantiation or Usage | Method Name | Citation
--- | --- | --- 
`pygrank.algorithms.oversampling.SeedOversampling(ranker)` | SeedO | krasanakis2019boosted
`pygrank.algorithms.oversampling.BoostedSeedOversampling(ranker)` | SeedBO | krasanakis2019boosted
`pygrank.algorithms.pagerank.PageRank(converge_to_eigenvectors=True)` | VenueRank | krasanakis2018venuerank
`G = pygrank.postprocess.fairness.to_fairwalk(G, sensitive)` | FairWalk |rahman2019fairwalk
`pygrank.algorithms.postprocess.fairness.FairPostprocessor(ranker,'O')` | LFPRO | tsioutsiouliklis2020fairness
`pygrank.algorithms.postprocess.fairness.FairPersonalizer(ranker)` | FP | krasanakis2020fairconstr
`pygrank.algorithms.postprocess.fairness.FairPersonalizer(ranker,0.8)` | CFP | krasanakis2020fairconstr
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
  title={VenueRank: Identifying Venues that Contribute to Artist Popularity.},
  author={Krasanakis, Emmanouil and Schinas, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Yiannis and Mitkas, Pericles A},
  booktitle={ISMIR},
  pages={702--708},
  year={2018}
}
```


### Under Review
```
@unpublished{krasanakis2020stopping,
  title={Stopping Personalized PageRank without an Error Tolerance Parameter},
  author={Krasanakis, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Ioannis},
  year={2020},
  note = {unpublished}
}
```

```
@unpublished{krasanakis2020fairconstr,
  title={Applying Fairness Constraints on Graph Node Ranks under Personalization Bias},
  author={Krasanakis, Emmanouil and Papadopoulos, Symeon and Kompatsiaris, Ioannis},
  year={2020},
  note = {unpublished}
}
```

### Related
Here, we list additional publications whose methods are implemented in this library.
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