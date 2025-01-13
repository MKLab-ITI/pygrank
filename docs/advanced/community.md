# Community Detection

`pygrank` provides the capability of automatically selecting the best algorithm among a list
of candidate ones. This can be achieved either with supervised measures (typically AUC)
for which a validation subset of nodes is withheld to determine the best algorithm
or with unsupervised measures. Then, algorithms can be used to automatically detect communities.

First, let us load a dataset with several communities, create training-test splits of those
and used the package to create a bunch of (normalized) graph filters.

```python
import pygrank as pg
_, graph, communities = next(pg.load_datasets_multiple_communities(["EUCore"], max_group_number=3))
train, test = pg.split(communities, 0.05)  # 5% of community members are known
algorithms = pg.create_variations(pg.create_demo_filters(), pg.Normalize)
```

We can now define two graph filters that automtatically select the best method with an
AUC-based supervised evaluation and a modularity-based unsupervised evaluation respectively.
In the second case, we set `fraction_of_training=1` to not withhold validation data from
the compared node ranking algorithms. It is important to note that algorithm normalization
helps modularity create comparable assessments between different algorithms.

```python
supervised_algorithm = pg.AlgorithmSelection(algorithms.values(), measure=pg.AUC)
modularity_algorithm = pg.AlgorithmSelection(algorithms.values(), fraction_of_training=1, measure=pg.Modularity().as_supervised_method())
```

We now run the algorithms for all communities to report the average AUC on the test sets.
Testing is made to exclude original seed nodes to ensure that results are not biased.

```python
supervised_aucs = list()
modularity_aucs = list()
for seeds, members in zip(train.values(), test.values()):
    measure = pg.AUC(members, exclude=seeds)
    supervised_aucs.append(measure(supervised_algorithm(graph, seeds)))
    modularity_aucs.append(measure(modularity_algorithm(graph, seeds)))

print("Supervised", sum(supervised_aucs) / len(supervised_aucs))
print("Modularity", sum(modularity_aucs)/len(modularity_aucs))
```

For very few seed nodes (in this case, they are fewer than 5 in tested communities)
supervised evaluation lacks robustness, especially since removing seeds shaves off
a large portion of information from node ranking algorithms. This can be seen in 
the above example, where modularity-based algorithm selection outperformed the
supervised one. On the other hand, for more nodes in the training seeds (such as
10% of total community members) supervised evaluation is the better one.

