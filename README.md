# pygrank
Recommendation algorithms for large graphs.

*Dependencies: sklearn, scipy, networkx*

## Usage
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

## Notes
This is a not-yet clean and unoptimized version, with many residual prints
and no documentation details that is used in our research. 

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
