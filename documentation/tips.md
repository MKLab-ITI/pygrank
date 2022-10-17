<center><h1>:bulb: Big Data FAQ</h1></center>
This documents provides a FAQ on how to handle big graphs, with millions
or more of edges.

### Which backend to use?
Use *numpy* (the default backend). It's the most well-tested
and memory efficient.

### Which algorithm to use?
`pygrank` algorithms are split into graph filters and postprocessors
to augment their outcome. Here we touch on graph filters. 
If you don't know anything else about your data, the following
graph filter is recommended:
```python
import pygrank as pg

algorithm = pg.PageRank(alpha=0.9, tol=1.E-9, max_iters=1000)
```
For historical reasons (e.g. compatibility with `networkx`), 
these are not the default parameters values.
But they tend to work well in big graphs. A little explanation on 
the choices:
- Personalized PageRank is equivalent to stochastic random
walks with average length *1/(1-alpha)* hops away from
seed nodes. 
- At the same time, you need a small enough 
numerical tolerance to make sure that your numer of seeds
divided by your number of nodes is not immediately
smaller than that.
- Higher diffusion parameters (*alpha*) and
lower numerical tolerances drastically increase the number of
iterations it takes for recursive graph filters to converge.
Thus, a higher cap to the computational limit should be
placed to make sure that this is not exceeded before 
convergence. (Note: Never run algorithms with computational
limits higher than your allocated computational budget, 
as a last failsafe against unforeseen algorithmic properties.)

### My communities do not comprise enough members.
Try to increase the receptive field of node ranking algorithms,
for example by increasing *alpha* in pagerank. If you have increased
the receptive field but require more expansions try applying
the `SeedOversampling()` and, if you are fine with its computational
demands, `BoostedSeedOversampling()` postprocessors on your 
algorithms.


### My graph is already a scipy sparse matrix.
Note that node ranking algorithms and graph signals
typically require graphs. However, sometimes
it is more computationally efficient to construct
and move around sparse scipy adjacency matrices, 
for example to avoid additional memory allocations.

For these situations, given an adjacency matrix
*adj*, you can convert it to a graph wrapping its data
in *O(1)* time and memory with the following code:
```python
import pygrank as pg

adj = ...  # a square sparse scipy array
graph = pg.AdjacencyWrapper(adj, directed=True)
```
In this case, the graph's nodes are considered to be
the numerical values *0,1,..,adj.shape[0]-1*.
The *directed* argument in the constructor only
affects the type of *"auto"* normalization in
preprocessors.
