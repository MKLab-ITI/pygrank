# Graph Filters

Graph filters are algorithms that spread the node values stored in graph signals
through graphs by diffusing them through edges for several hops of different weights.
This process produces new graph signals. The original graph signal is called 
the *personalization*, and often
indicates the likelihood that nodes have a certain property, such as being members
of a structural or metadata community. Graph filters refine these
initial estimates by providing improved (probability) scores for all nodes.
Below we present a typical node ranking pipeline that starts from a known personalization,
applies a graph filter, potentially postprocesses its outcome, and eventually arrives at new node values.

<img src="../pipeline.png" alt="pipeline" style="width: 60%;">

Filters are created based on a constructor that takes as input several keyword
arguments affecting how they work. An exhaustive list of ready-to-use graph filters 
and their constructors
found [here](../generated/filters.md).
More complicated node ranking algorithms can be obtained by applying postprocessors on
filters. This is covered in the [next section](postprocessors.md).
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
[here](../advanced/graph_preprocessing.md).

We also stop the algorithm at numerical
tolerance *1.E-9*. Smaller tolerances are more accurate in exactly solving
each algorithm's exact outputs but take longer to converge. Since this is
a particularly hard graph to rank despite being very small (pygrank throws an exception if we stick
with the default number of 100 iterations), we increase the budget for iterations
to *2000*. An advanced
discussion on convergence management strategies is presented [here](../advanced/convergence.md).

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
