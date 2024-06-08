# The Basics

At the core of `pygrank` lies the concept of *graph signals*, which map graph nodes to numerical scores. 
Supervised and unsupervised measures evaluate the predictive/ranking quality
of graph signals. The library's main purpose
is to define and efficiently run node ranking algorithms. These start from *graph filters*, 
which iteratively diffuse the scores of nodes to neighbors that are connected to them. 
The output of filters can be processed with additional components.
Below we present a typical node ranking pipeline that starts from a known personalization,
applies a graph filter, potentially postprocesses its outcome, and eventually arrives at new node values.

<img src="../pipeline.png" alt="pipeline" style="width: 60%;">

## Graph Signals

A *graph signal* is a way to organize numerical values 
that correspond to the nodes of a graph. Signals are used as the
inputs and outputs of node ranking algorithms, though calls to
the latter are for convenience overloaded to automatically 
construct signals if different arguments are provided.
Here is how to create a simple signal that includes nodes
'A' and 'C' with values of 3 and 2 respectively
sets 0 to all other nodes:

```python
import pygrank as pg
import networkx as nx

graph = nx.Graph()
graph.add_edge('A', 'B')
graph.add_edge('A', 'C')
graph.add_edge('C', 'D')
graph.add_edge('D', 'E')
signal = pg.to_signal(graph, {'A': 3, 'C': 2})

print(signal['A'], signal['B'])  # 3.0 0.0
```

In general, graph signals can be constructed with the
expression, `pg.to_signal(graph, obj)` that takes
as input a graph and some data. These data can be
in various convenient formats listed below:

| Format                                                    | Description                                                                                                                                                                                        | Example data                 |
|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------|
| Maps of node values                                       | Assume all other missing elements to represent zero values.                                                                                                                                        | `obj={'A': 3, 'C': 2}`       |
| Numpy arrays, lists or tensors.                           | Represent numerical values for each graph node, where nodes are organized per their traversal order in the graph's iterator. If tensors are provided, most computations remain backpropagate-able. | `obj=np.array([3, 0, 2, 0])` |
| List or set of nodes with fewer elements than graph nodes | Assigns value of one to specified nodes. Other nodes obtain zero values.                                                                                                                           | `obj=['A', 'C']`             |
| `None`                                                    | Interpreted as a signal of ones.                                                                                                                                                                   | `obj=None`                   |


An internal representation of signal
values in the second of the above formats
can be accessed through the `signal.np` attribute. 
Out-of-the-box, your running backend will be 
`"numpy"` and therefore this will also be numpy array.
However,
different data types may be held, depending on the *current*
backend; switching backends after a signal is defined
or computed will yield
a representation in the new backend's preferred format. 

Arithmetic operations defined by the running backend
are also directly applicable to signals by implying the `np` attribute,
like below. All operations involving signals should occur on the same
graph, and include a respective assertion.

```python
signal = signal / pg.sum(signal)
print([(k,v) for k,v in signal.items()])  # [('A', 0.6), ('B', 0.0), ('C', 0.4), ('D', 0.0), ('E', 0.0)]
```

## Graph Filters

Graph filters are algorithms that spread the node values stored in graph signals
through graphs by diffusing them through edges for several hops of different weights.
This process produces new graph signals. The original graph signal is called 
the *personalization*, and often
indicates the likelihood that nodes have a certain property, such as being members
of a structural or metadata community. Graph filters refine these
initial estimates by providing improved (probability) scores for all nodes.

Filters are created based on a constructor that takes as input several keyword
arguments affecting how they work. An exhaustive list of ready-to-use graph filters 
and their constructors
found [here](../generated/graph_filters.md).
More complicated node ranking algorithms can be obtained by applying postprocessors on
filters. This is covered [later](#postprocessors).
After its initialization, a filter `alg` can run
with one of the following three patterns (the first two are interchangeable):

| Pattern                                              | Description                                                                                                                                               |
|------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `scores = alg(pg.to_signal(graph, personalization))` | More explicit. Enables some rarely used advanced options.                                                                                                 |
| `scores = alg(graph, personalization)`               | Internally calls the `to_signal` method. Faster code writing.                                                                                             |
| `scores = alg(graph)`                                | Computes non-personalized node scores by setting the personalization value to 1 for each node. This turns the algorithm's outcome into centrality scores. |

As an example, let us define an personalized PageRank filter. If the personalization is
binary (i.e. all nodes have initial scores either 0 or 1) this algorithm
is equivalent to a stochastic Markov process where it starts from the nodes
with initial scores 1, iteratively jumps to neighbors randomly, and has
a fixed probability *1-alpha* to restart. Node scores capture the probabilities 
of arriving at each node.

We use a restart probability at each step *1-alpha=0.01* and will
perform "col" (column-wise) normalization of the adjacency matrix to make
jumps to neighbors have equal probabilities. The alternative is "symmetric"
normalization, where the probabilities of moving between two nodes are the
same for both movement directions. Without an argument, the type of
normalization is selected based on whether the graph is directed or undirected
respectively. Find more in the advanced graph preprocessing section 
[here](preprocessing.md).

We also stop the algorithm at numerical
tolerance *1.E-9*. Smaller tolerances are more accurate in exactly solving
each algorithm's exact outputs but take longer to converge. Since this is
a particularly hard graph to rank despite being very small (pygrank throws an exception if we stick
with the default number of 100 iterations), we increase the budget for iterations
to *2000*. An advanced
discussion on convergence management strategies is presented [later](#convergence).

```python
import pygrank as pg
import networkx as nx

graph = nx.Graph()
graph.add_edge('A', 'B')
graph.add_edge('A', 'C')
graph.add_edge('C', 'D')
graph.add_edge('D', 'E')
signal = pg.to_signal(graph, {'A': 3, 'C': 2})

algorithm = pg.PageRank(alpha=0.99, normalization="col", tol=1.E-9, max_iters=2000)
```

!!! info
    Filters like pagerank focus on diffusing scores fewer hops away
    and are thus low-pass in that they reduce graph adjacency matrix's eigenvalues,
    which are often considered the spectrum.
    In practice, they smoothen the personalization through the graph's structure.

Having defined this algorithm, we now pass a graph signal through the pipeline below.
For the time being, we do not perform any postprocessing and rely on the base filter.
Notice that both 'A' and 'C' end up with the higher scores,
which are approximately 0.25. 'D' forms a circle with these
in the graph's structure and thus, by merit of being structurally close,
is scored closely to these two as 0.24. Finally, the other two nodes
assume lower values. Essentially, we obtain the structural relatedness 
of various nodes to the personalization:


```python
scores = algorithm(signal)  # or algorithm(graph, {'A':1, 'C': 2})
print(scores)  # [('A', 0.25613418536078547), ('B', 0.12678642237010243), ('C', 0.2517487443382047), ('D', 0.24436832596280528), ('E', 0.12096232196810223)]
```


!!! info
    In the above code, we could also pass the graph and
    dictionary `{'A':1, 'C': 2}` as positional arguments in place
    of the signal, and the latter would be generated internally.
    For signals we can omit the graph argument because a graph
    is already tied to the signal.

## Postprocessors

Filter outcomes of graph often require additional processing steps, for example to perform
normalization, improve their quality, or apply fairness constraints.
Postptprocessors wrap base filters to improve their outcomes, and the result
node ranking algorithms are called as if they were still filters. The filters
can either be supplied to construcors, or the postprocessors may be initialized
from the rest of their arguments and applied onto filters afterwards with the 
functional chain pattern `algorithm = filter >> postprocessor`.


A list of ready-to-use postprocessors can be
found [here](../generated/postprocessors.md). Simpler ones perform
normalization, for example to enforce the maximal or the sum 
of node scores to be 1. There also exist thresholding schemes, which can be used
for binary community detection, as well as methods to make node
comparisons non-parametric by transforming scores to ordinalities.

More complex postprocessing mechanisms involve re-running the 
base filters with augmented personalization. This happens both for
seed oversampling postprocessors, which aim to augment node scores
by providing more example nodes, and for fairness-aware posteriors,
which aim to make node scores adhere to some fairness constraint, 
such as disparate impact.

Let us consider a simple toy scenario where we want the graph signal outputted
by a filter to always be normalized so that its largest node score is one. For
graph `G`, signal `signal` and filter `alg`, 
and can use the postprocessor `Normalize("max")`. For convenience, simpler postprocessors 
like this one supply a method to transform graph signals like so:

```python
scores = alg(graph, signal)
normalized_scores = pg.Normalize("max").transform(scores)
print(list(normalized_scores.items()))
# [('A', 1.0), ('B', 0.4950000024069947), ('C', 0.9828783455187619), ('D', 0.9540636897749238), ('E', 0.472261528845582)]
```

The pattern that works for **all** postprocessors
is to wrap base algorithms, like in the following equivalent example:


```python
nalg = alg >> pg.Normalize("max")  # also valid pg.Normalize("max", alg) or pg.Normalize(alg, "max")
nscores = nalg(graph, signal)
print(nscores)
# [('A', 1.0), ('B', 0.4950000024069947), ('C', 0.9828783455187619), ('D', 0.9540636897749238), ('E', 0.472261528845582)]
```

We can add more steps, such as
an element-wise exponential transformation of scores
before normalization:

```python
nealg = alg >> pg.Transformer(pf.exp) >> pg.Normalize("max")
nescores = nealg(graph, signal)
print(nescores)
# [('A', 1.0), ('B', 0.8786683440755908), ('C', 0.9956241609824301), ('D', 0.9883030876536782), ('E', 0.8735657648099558)]
```

## Convergence

All graph filter constructors have a `convergence` argument that
indicates an object to help determine their convergence criteria, such as type of
error and tolerance for numerical convergence. If no such argument is passed
to the constructor, a `pygrank.ConvergenceManager` object
is automatically instantiated by borrowing whichever extra arguments it can
from those passed to algorithm constructors. These arguments can be:

- *tol:* Indicates the numerical tolerance level required for convergence (default is 1.E-6).
- *error_type:* Indicates how differences between two graph signals are computed. The default value is `pygrank.Mabs` but any other supervised [measure](../userguide/evaluation.md) that computes the differences between consecutive iterations can be used. The string "iters" can also be used to make the algorithm stop only when max_iters are reached (see below).
- *max_iters:* Indicates the maximum number of iterations the algorithm can run for (default is 100). This quantity works as a safety net to guarantee algorithm termination. 

Sometimes, it suffices to reach a robust node rank order instead of precise 
values. To cover such cases we have implemented a different convergence criterion
``RankOrderConvergenceManager`` that stops 
at a robust node order [krasanakis2020stopping]. This criterion is specifically intended to be used with PageRank 
as the base ranking algorithm and needs to know that algorithm's diffusion
rate ``alpha``, which is passed as its first argument.

```python
import pygrank as pg

G, personalization = ...
alpha = 0.85
ranker = pg.PageRank(alpha=alpha, convergence=pg.RankOrderConvergenceManager(alpha))
ordered_ranker = ranker >> pg.Ordinals()
ordered_ranks = ordered_ranker(G, personalization)
```

!!! info
    Since the node order was deemed more important than the specific rank values,
    a postprocessing step was added. 
