# pygrank
Recommendation algorithms for large graphs.

## Table of Contents
* [Table of Contents](#table-of-contents)
* [Installation](#installation)
* [Usage](#usage)
    + [Ranking Algorithms](#ranking-algorithms)
    + [Adjacency Matrix Normalization](#adjacency-matrix-normalization)
    + [Convergence Criteria](#convergence-criteria)
    + [Improving Ranking Outcome](#improving-ranking-outcome)
    + [Evaluation](#evaluation)
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

### Convergence Criteria
Run a PageRank algorithm and make it converge to a robust node order
```python
import networkx as nx
from pygrank.algorithms.pagerank import PageRank as Ranker
from pygrank.algorithms.utils import RankOrderConvergenceManager

G = nx.Graph()
seeds = list()
... # insert graph nodes and select some of them as seeds (e.g. see tests.py)
alpha = 0.85

algorithm = Ranker(alpha=alpha, convergence=RankOrderConvergenceManager(alpha))
ranks = algorithm.rank(G, {v: 1 for v in seeds})
```

:bulb: Since the node order is more important than the specific rank values,
a post-processing step to map nodes to that order can be added to the algorithm as:

```python
from pygrank.algorithms.pagerank import PageRank as Ranker
from pygrank.algorithms.utils import RankOrderConvergenceManager
from pygrank.algorithms.postprocess import Ordinals

...
alpha = 0.85
algorithm = Ranker(alpha=alpha, convergence=RankOrderConvergenceManager(alpha))
algorithm = Ordinals(algorithm)
...
```


### Improving Ranking Outcome
PageRank with seed oversampling
```python
import networkx as nx
from pygrank.algorithms.pagerank import PageRank as Ranker
from pygrank.algorithms.oversampling import SeedOversampling as Oversampler

G = nx.Graph()
seeds = list()
... # insert graph nodes and select some of them as seeds (e.g. see tests.py)

algorithm = Oversampler(Ranker(alpha=0.85, tol=1.E-6, max_iters=100)) # these are the default values
ranks = algorithm.rank(G, {v: 1 for v in seeds})
```


###### Hash the outcome of graph normalization to speed up multiple calls to the same graph
```python
import networkx as nx
from pygrank.algorithms.pagerank import PageRank as Ranker
from pygrank.algorithms.utils import preprocessor

G = nx.Graph()
seeds1 = list()
seeds2 = list()
... # insert graph nodes and select some of them as seeds (e.g. see tests.py)

pre = preprocessor(normalization="col", assume_immutability=True)
algorithm = Ranker(alpha=0.8, to_scipy=pre)
ranks = algorithm.rank(G, {v: 1 for v in seeds1})
ranks = algorithm.rank(G, {v: 1 for v in seeds2}) # does not re-compute the normalization
```

:bulb: Now preprocessor arguments can also be passed to the constructors of ranking algorithms.
This will make the ranking algorithm create its own preprocessor with the given arguments.

```python
import networkx as nx
from pygrank.algorithms.pagerank import PageRank as Ranker

G = nx.Graph()
seeds1 = list()
seeds2 = list()
... # insert graph nodes and select some of them as seeds (e.g. see tests.py)

algorithm = Ranker(alpha=0.8, normalization="col", assume_immutability=True)
ranks = algorithm.rank(G, {v: 1 for v in seeds1})
ranks = algorithm.rank(G, {v: 1 for v in seeds2}) # does not re-compute the normalization
```

:warning: If the normalization is not specified, it is set to "auto", which performs
"symmetric" normalization for undirected graphs and "col" normalization for directed ones.

### Evaluation

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
The following methods can be used to improve a base ranking algorithm `ranker`.

Instantiation | Method Name | Citation
--- | --- | --- 
`pygrank.algorithms.oversampling.SeedOversampling(ranker)` | Seed O | krasanakis2019boosted
`pygrank.algorithms.oversampling.BoostedSeedOversampling(ranker)` | Seed BO | krasanakis2019boosted
`pygrank.algorithms.postprocess.fairness.FairPostprocessor(ranker,'O')` | LFPRO | tsioutsiouliklis2020fairness
`pygrank.algorithms.postprocess.fairness.FairPersonalizer(ranker)` | FP | krasanakis2020fairconstr
`pygrank.algorithms.postprocess.fairness.FairPersonalizer(ranker,0.8)` | CFP | krasanakis2020fairconstr

### Publications
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