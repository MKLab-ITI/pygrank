# pygrank
Recommendation algorithms for large graphs.

*Dependencies: sklearn, scipy, networkx*

## Usage
###### How to run a PageRank algorithm
```python
import networkx as nx
from algorithms.pagerank import PageRank as Ranker
from algorithms.oversampling import SeedOversampling as Oversampler

G = nx.Graph()
seeds = list()
... # insert graph nodes and select some of them as seeds (e.g. see tests.py)

algorithm = Oversampler(Ranker(alpha=0.99))
ranks = algorithm.rank(G, {v: 1 for v in seeds})
```

###### How to evaluate with an unsupervised metric
```python
import algorithms.postprocess
import metrics.unsupervised

G, ranks = ... # calculate as per the first example
normalized_ranks = algorithms.postprocess.Normalize().rank(ranks)

metric = metrics.unsupervised.Conductance(G)
print(metric.evaluate(normalized_ranks))
```

###### How to evaluate with a supervised metric
```python
import metrics.supervised

G, seeds, algorithm = ... # as per the first example
seeds, ground_truth = metrics.utils.split_groups(seeds, fraction_of_training=0.5)

metrics.utils.remove_group_edges_from_graph(G, ground_truth)
ranks = algorithm.rank(G, {v: 1 for v in seeds})

metric = metrics.supervised.AUC({v: 1 for v in ground_truth})
print(metric.evaluate(ranks))
```

###### How to evaluate multiple ranks
```python
import networkx as nx
from algorithms.pagerank import PageRank as Ranker
from algorithms.postprocess import Normalize as Normalizer
from algorithms.oversampling import BoostedSeedOversampling as Oversampler
import metrics.supervised
import metrics.unsupervised
import metrics.multigroup

# Construct data
G = nx.Graph()
groups = {}
groups["group1"] = list()
... 

# Split to training and test data
training_groups, test_groups = metrics.utils.split_groups(groups)
metrics.utils.remove_group_edges_from_graph(G, test_groups)

# Calculate ranks and put them in a map
algorithm = Normalizer(Oversampler(Ranker(alpha=0.99)))
ranks = {group_id: algorithm.rank(G, {v: 1 for v in group}) 
        for group_id, group in training_groups.items()}
        

# Evaluation with Conductance
conductance = metrics.multigroup.MultiUnsupervised(metrics.unsupervised.Conductance, G)
print(conductance.evaluate(ranks))

# Evaluation with LinkAUC
link_AUC = metrics.multigroup.LinkAUC(G)
print(link_AUC.evaluate(ranks))

# Evaluation with AUC
auc = metrics.multigroup.MultiSupervised(metrics.supervised.AUC, metrics.utils.to_seeds(test_groups))
print(auc.evaluate(ranks))
        
```


## References
###### *OversamplingRank*, *BoostingRank*
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
