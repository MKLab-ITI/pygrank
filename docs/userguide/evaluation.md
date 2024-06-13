# Evaluation

`pygrank` provides a wide breadth of measures
that can assess node ranking algorithms, as well as comprehensive
code interfaces with which to quickly set up experiments at scale to compare algorithms.
An offshoot of these capabilities is autotuning, which is very useful in practice
and covered [here](autotuning.md).

## Measures

Provided measures are mainly a) supervised in that they compare graph signal posteriors
(the node scores outputted by node ranking algorithms)
with some known ground truth ones, or b) unsupervised in that they assess whether 
posteriors satisfy a desired property, such as low conductance or density.
These two types of measures are instantiated with keyword parameters.
A list of measures can be
found [here](../generated/measures.md). We have not addressed some types yet; look for measures
that extend the `Supervised` and `Unsupervised` base classes. After initialization,
measures are callables that output a numeric value given graph signal posteriors.

!!! info
    Supervised measures typically admit two arguments:
    
    - *known_scores*: This is a mandatory first argument that holds
    a graph signal or data that together with a graph can automatically instantiate
    a signal. It indicates ground truth information considered as the desired posteriors.
    - *exclude*: Optional. A binary graph signal (or, again, data to extrapolate a graph signal)
    that indicates which nodes to exclude from evaluation. Typically, you
    will want to exclude all the nodes with non-zero values in the personalization,
    either with by providing the personalization signal or the list of nodes generating it.
    If None (default) all nodes are included in the evaluation. You can overwrite this value
    at any point with `measure.exclude = ...`.

    Unsupervised measures may also take an optional *graph* argument. This helps convert
    provided data to a graph signal internally when calling the measure with not a signal.

An example of how to use measures follows. In that we also
used `Threshold` postprocessor to convert scores to binary values,
but measures also accept non-crisp scores that have intermediate values;
in this case stochastic equivalents are computed.


```python
import pygrank as pg
import networkx as nx

graph = nx.Graph()
graph.add_edge('A', 'B')
graph.add_edge('A', 'C')
graph.add_edge('C', 'D')
graph.add_edge('D', 'E')
signal = pg.to_signal(graph, {'A': 3, 'C': 2})

algorithm = pg.HeatKernel(t=5) >> pg.Normalize("max") >> pg.Threshold(0.5)
scores = algorithm(signal)
print(scores)  # {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0, 'E': 0.0}

supervised = pg.Accuracy({'D': 1}, exclude=signal)  #  or exclude=['A', 'C']
print(supervised(scores))  # 0.6666666666666667

unsupervised = pg.Density()  # lower is better
print(unsupervised(scores))  # 0.5
```


In addition to base measures, there also exist c) variants
that combine them, and d) evaluation that runs on multiple communities.
To begin with, multiple measures can be aggregated through the `pg.AM` and 
`pg.GM` classes, which respectively perform arithmetic and geometric
averaging of assessment values. After instantiateing these
classes, measures can be added to them with a chain of `.add` method calls 
that also accept optional weights and saturation
thresholds. Here is an example:

```python
import pygrank as pg

known_scores, algorithm, personalization, sensitivity_scores = ...
auc = pg.AUC(known_scores, exclude=personalization)
prule = pg.pRule(sensitivity_scores, exclude=personalization)
measure = pg.AM(differentiable=True).add(auc).add(prule, weight=10., max_val=0.8)
```

!!! info
    When working on autodifferntiation backends,
    the constructor of the measure combination mechanism
    can be annotated with `differentiable=True`. This indicates that,
    when thresholds are provided, a differentiable relaxation of saturation 
    should be applied instead of a simple min or max. 

Extension of base measures to multiple communities can be obtained from the
`pg.MultiSupervised` and `pg.MultiUnsupervised` classes. These take as arguments
the *classes* of base measures and any keyword arguments
needed to construct them. In case of the *exclude* attribute of
supervised measures a dictionary from community
identifiers (like strings with community names) to
graph signals/signal data should be provided to the `pg.MultiSupervised`. 
When called, these measures process and output dictionaries from community
identifiers to create an assessment value for each of those communities.
Values are also returned as a similar dictionary. Practical code
using these classes may look like this:

```python
import pygrank as pg

_, graph, communities = next(pg.load_datasets_multiple_communities(["dblp"]))
algorithm = pg.PageRank(alpha=0.9, assume_immutability=True)  # cache graph preprocessing
algorithm = algorithm >> pg.Normalize("max")
comm_scores = {name: algorithm(graph, members) for name, members in communities.items()}

measure = pg.MultiUnsupervised(pg.Conductance)
print(measure(comm_scores))
```

In case of unsupervised evaluation 
you may also be interested in computing link-based
unsupervised measures. This is done with the
`pg.LinkAssessmen` class; this needs the graph
as a constructor argument and can only be computed
when multiple communities are analysed together.
However, it often performs a qualitative evaluation
closer to supervised assessment [krasanakis2020unsupervised].
Continuing from the
previous snippet, evaluation of scores with this
strategy can be performed like so:

```python
import tqdm  # install this to be able to set it as a progress bar argument below

measure = pg.LinkAssessment(graph, progress=tqdm.tqdm)  # graph argument is mandatory
print(measure(comm_scores))
```

!!! info
    Link-based evaluation or, in general,
    some kind of unsupervised evaluation is needed 
    when there are only a few community
    members that serve as examples towards which to
    score proximity. In this case, splitting example members to
    training and test sets risks degrading supervised
    evaluation quality.


## Datasets

`pygrank` provides a variety of datasets to be automatically downloaded
and imported. 
Messages in the error console point to respective dataset sources.
You can also add your own datasets by placing them in a visible
directory (a `data` directory either under the project or under `home/.pygrank/data`
where automatically downloaded datasets are stored).
Please refer to the repository of datasets
[generated for pygrank](https://github.com/maniospas/pygrank-datasets) for
a description of conventions needed to create new datasets; broadly, these follow
the format of the SNAP repository and include pairs.txt and groups.txt file.

A summary of all datasets which are automatically downloaded by the project 
and their available types of data can 
be found [here](../generated/datasets.md). A lists of all dataset names can be obtained 
programmatically with the method `downloadable_small_datasets()`, but 
you can also use `downloadable_small_datasets()` to get a list of dataset names
on datasets that run fastly (within seconds instead of minutes or tens of minutes)
in modern machines.
Lists of dataset names like these are provided to methods that
iteratively yield loaded data. Five methods with the ability to load
different amounts of information are provided below. For each method,
datasets lacking needed information are silently omitted.

| Method                               | Description                                                                                                                                                                                                |
|--------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `load_one`                           | Gets as an input one dataset name and returns a tuple of that name, the graph, and the first group found by `load_datasets_one_community`.                                                                 |
| `load_datasets_graph`                | Yields respective dataset graphs.                                                                                                                                                                          |
| `load_datasets_one_community`        | Yields tuples of dataset names, graphs and node lists, where the lists correspond to one of the (structural or metadata) communities of graph nodes.                                                       |
| `load_datasets_all_communities`      | Yields the same tuples as before, but also traverses all possible datasets node communities (thus, there are many loading outcomes for each dataset). Community identifiers are appended to dataset names. |
| `load_datasets_multiple_communities` | Yields tuples of dataset names, graphs and hashmaps, where the latter map community identifiers to lists of nodes.                                                                                         |
| `load_feature_dataset`               | Yields respective tuples of graphs, features and node labels, where the last two are organized into numpy arrays whose lines correspond to graph nodes (based on the order the graphs are traversed).      |

All dataset loading methods admit the following list of keyword parameters:

- *path*: This is directory location (default is `"data"`) to seach for datasets and download them in. 
Take care to *not* add a trailing slash.
- *min_group_size:* The minimum number of nodes for each group to accept (default is 0.01). Groups
with fewer nodes are ignored. If a value less than 1 is provided, it is multiplied with the number of graph nodes.
- *min_group_id:* The
- *max_group_number:* The maximum number of communities to load; stops at this number. Default is 20.
- *prepend_all_nodes:* If True, creates a first community that holds all graph nodes. Default is False.
    graph_api=nx,

 The following code can be used to load datasets for overlapping community detection
given that each node community should be experimented on separately. 

```python
import pygrank as pg

datasets = pg.downloadable_small_datasets()
print(datasets)  # ['bigraph', 'blockmodel', 'citeseer', 'eucore', 'graph5', 'graph9']

for dataset, graph, group in pg.load_datasets_all_communities(datasets):
    print(dataset, ":", len(group), "community members", len(graph), "nodes",  graph.number_of_edges(), "edges")

# REQUIRED CITATION: Please visit the url https://github.com/maniospas/pygrank-datasets for instructions on how to cite the dataset bigraph in your research
# bigraph0 : 300 community members 600 nodes 27092.0 edges
# bigraph1 : 300 community members 600 nodes 27092.0 edges
# ...
```




## Benchmarks

`pygrank` offers the ability to conduct benchmark experiments that compare
many node ranking algorithms and parameters on a wide range of graphs. 
A simple way to obtain some fastly-running algorithms and small datasets and
compare them under the AUC measure would be per:

```python
import pygrank as pg

dataset_names = pg.downloadable_small_datasets()
print(dataset_names)  # ['citeseer', 'eucore']

algorithms = pg.create_demo_filters()
print(algorithms.keys())  # dict_keys(['PPR.85', 'PPR.9', 'PPR.99', 'HK3', 'HK5', 'HK7'])

loader = pg.load_datasets_one_community(dataset_names)
pg.benchmark_print(pg.benchmark(algorithms, loader, pg.AUC))

#                	 PPR.85  	 PPR.9  	 PPR.99  	 HK3  	 HK5  	 HK7 
# citeseer       	 .89     	 .89    	 .89     	 .89  	 .89  	 .89 
# eucore         	 .85     	 .71    	 .71     	 .91  	 .89  	 .83 
# graph9         	 1.00    	 1.00   	 1.00    	 1.00 	 1.00 	 1.00
# bigraph        	 .96     	 .77    	 .77     	 1.00 	 .98  	 .86 
# REQUIRED CITATION: Please visit the url https://linqs.soe.ucsc.edu/data for instructions on how to cite the dataset citeseer in your research
# REQUIRED CITATION: Please visit the url https://snap.stanford.edu/data/email-Eu-core.html for instructions on how to cite the dataset eucore in your research
# REQUIRED CITATION: Please visit the url https://github.com/maniospas/pygrank-datasets for instructions on how to cite the dataset graph5 in your research
# REQUIRED CITATION: Please visit the url https://github.com/maniospas/pygrank-datasets for instructions on how to cite the dataset graph9 in your research
# REQUIRED CITATION: Please visit the url https://github.com/maniospas/pygrank-datasets for instructions on how to cite the dataset bigraph in your research
```

You can apply a postprocessor to a dictionary of node ranking algorithms like below.
You can apply several postprocessors at once, for example if the end-goal is to compare between them.
When apply prostprocessors an masse, a prefix is declared to prepend to algorithm names 
(e.g., the HK5Sweep variation of HK5 will be created).

```python
sweeps = pg.create_variations(algorithms, {"Sweep": pg.Sweep})
algorithms = algorithms | sweeps  # merge old and new algorithms
```

For massive experimentation, generate many algorithms and variations with the following
methods:

```python
postprocessors = pg.create_many_variation_types()
filters = pg.create_many_filters()
algorithms = pg.create_variations(filters, postprocessors)
```

Specific algorithms can also be added
or used in place of the `algorithms` dictionary.
For example, add an automatically-tuned
algorithm (more on these later) with default parameters per the following code and
then re-run experiments to see how well it fares..

```python
algorithms["Tuned"] = pg.ParameterTuner()
```

!!! warning
    To run a new series of experiments, the loader needs to be called anew (it is an iterator).
