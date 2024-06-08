# Graph Preprocessing

## Normalization

Graph filters all use the same default graph normalization scheme
that performs symmetric (i.e. Laplacian-like) normalization 
for undirected graphs and column-wise normalization that
follows a true probabilistic formulation of transition probabilities
for directed graphs, such as `DiGraph` instances. The type of
normalization can be specified by passing a *normalization*
argument to constructors of ranking algorithms. This parameter 
can have the following values:

| Normalization | Description                                                                                                                                                                                     |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `"auto"`      | The above-described default behavior.                                                                                                                                                           |
| `"col"`       | Column-wise normalization.                                                                                                                                                                      |
| `"symmetric"` | Symmetric normalization.                                                                                                                                                                        |
| `"laplacian"` | Generates the Laplacian of the graph.                                                                                                                                                           |                                                                                                                                                             |
| `"salsa"`     | The row-wise normalization employed by the salsa algorithm.                                                                                                                                     | 
| `"none"`      | (A string with text "none".) Avoids any normalization, for example, because edge weights already hold the normalization.                                                                        |
| callable      | A callable applied to a `scipy` sparse adjacency matrix of the "numpy" backend (irrespective of the actually active backend). When applied, it ignores the preprocessor's *reduction* argument. |

Additionally, a *transform_adjacency* method can be provided
to modify the final adjacency matrix after all computations conclude.
This method runs within the currently active backend.
By default this is a tautology `lambda x: x`.
To create smoother versions of adjacency matrices,
*renormalization* argument may be provided
to add a multiple of the unit matrix to the adjacency matrix,
a concept called the renormalization trick.
This by default 0, but can help shrink the spectrum.For example,
you can create a strongly local version of the adjacency matrix like this:

```python
alg = Algorithm(renormalization=2)
```


In all cases, adjacency matrix normalization involves the
computationally intensive operation of converting the graph 
into a scipy sparse matrix each time node
ranking algorithms are called. `pygrank`
provides a way to avoid recomputing the normalization
during large-scale experiments by the same algorithm for 
the same graphs by passing an argument `assume_immutability=True`
to the algorithms's constructor, which indicates that
the the graph does not change between runs of the algorithm
and hence computes the normalization only once for each given
graph, a process known as hashing. 
Hashing only uses the Python object's hash method, 
so a different instance of the same graph will recompute the 
normalization if it points at a different memory location.


!!! warning
    Do not alter graph objects after passing them to
    calls of node ranking algorithms for the first time
    if you set `assume_immutability=True`. If altering the
    graph is necessary midway through your code, create a copy
    instance, for example with one of *networkx*'s in-built methods and
    edit that one.

Hashing the outcome of graph normalization to
speed up multiple calls to the same graph can be achieved
as per the following code:

```python
import pygrank as pg

graph, personalization1, personalization2 = ...
algorithm = pg.PageRank(alpha=0.85, normalization="col", assume_immutability=True)
ranks1 = algorithm(graph, personalization1)
ranks2 = algorithm(graph, personalization2) # does not re-compute the normalization
```

## Caching

Sometimes, many different algorithms are applied on the
same graph. In this case, to prevent each one
from recomputing the hashing already calculated by others,
they can be made to share the same normalization method. This 
can be done by using a shared instance of the 
normalization preprocessing `pg.preprocessor`, 
which can be passed as the `preprocessor` argument of ranking algorithm
constructors. In this case, the `normalization`, `renormalization` 
and `assume_immutability`
arguments should be passed to the preprocessor and will be ignored by the
constructors (what would otherwise happen is that the constructors
would create a prerpocessor with these arguments).

Basically, when the default value `preprocessor=None` is passed to ranking algorithm
constructors, these create a new preprocessing instance
with the `normalization`, `renormalization` and `assume_immutability`
values passed
to their constructor. These two arguments are completely ignored
if a preprocessor instance is passed to the ranking algorithm.
Direct use of these arguments without needing to instantiate a
preprocessor was demonstrated in the previous code example.

Using the same outcome of graph preprocessing 
to speed up multiple rank calls to the same graph by
different ranking algorithms can be done as:
```python
import pygrank as pg

graph, personalization1, personalization2 = ...
pre = pg.preprocessor(normalization="col", assume_immutability=True)
algorithm1 = pg.PageRank(alpha=0.85, preprocessor=pre)
algorithm2 = pg.HeatKernel(alpha=0.85, preprocessor=pre)
ranks1 = algorithm1(graph, personalization1)
ranks2 = algorithm2(graph, personalization2) # does not re-compute the normalization
```

!!! info
    When benchmarking in the above code you can call `pre(graph)`
    before the first `rank(...)` call to make sure that that call
    does not also perform the first normalization whose outcome will
    be hashed and immediately retrieved by subsequent calls.


## Cross-origin resources

Preprocessing admits a *cors* argument that toggles
an optimization for huge speedups when
constantly switching between backends at the
cost of additional memory and badly-defined behavior.
Default is False, which saves a lot of memory. However,
the maximum used memory when using *cors*
remains the same when processing one graph and
switching between up to two backends
out of which one is "numpy".

!!! warning 
    Enabling cors will not run certain parts of
    preprocessing, because it performs immediate backend switching
    with cached versions of the adjacency matrix. There is
    no error checking involved. For this reason:
    
    - Remember that the only preprocessor that will actually
    run computations is the first one that runs. If you
    are unsure, either set the same arguments or generate
    the preprocessor first and share it between all algorithms.
    - Make sure that `cors=True` holds for every
    preprocessor and graph filter that will use
    cors graphs.

    **YOU HAVE BEEN WARNED.**

If *cors* is enabled (set to True), backend primitives
holding the outcome of graph preprocessing are
enriched with additional private metadata that enable their
usage as base graphs when passing through other preprocessors 
in other backends. There is one main usage pattern:
obtaining an adjacency
matrix that represents the graph, but which is a thin wrapper
around fast-switching backend primitives. That is,
you need to preprocess your graph per:

```python
graph = pg.preprocessor(cors=True, ...)(graph)
```

This is technically an adjacency matrix but has
all the necessary ducktyping to be considered 
a graph by `pygrank`. If you plan to stay within
one backend, you can also run the same command
without enabling cors anywhere and
construct signals per normal with 
`pg.to_signal(graph, personalization_data)`.
However, enabling cors is mandatory
when creating or planning to use
graph signals with the adjacency
you defined at different backends.

Usefulness of cross-origin resources in demonstrated in the
following graph neural network example, where training
time is sped up by 25% when `cors=True` in the
constructor of the `GenericGraphFilter`:

```python
import pygrank as pg
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.regularizers import L2


class APPNP(tf.keras.Sequential):
    def __init__(self, num_inputs, num_outputs, hidden=64):
        super().__init__(
            [
                Dropout(0.5, input_shape=(num_inputs,)),
                Dense(hidden, activation="relu", kernel_regularizer=L2(0.005)),
                Dropout(0.5),
                Dense(num_outputs),
            ]
        )
        self.ranker = pg.ParameterTuner(
            lambda par: pg.GenericGraphFilter(
                [par[0] ** i for i in range(int(10))],
                error_type="iters",
                max_iters=10,
                cors=True,  # DON'T FORGET THIS
            ),
            max_vals=[1],
            min_vals=[0.5],
            verbose=False,
            measure=pg.Mabs,
            deviation_tol=0.01,
            tuning_backend="numpy",  # TUNING IN NUMPY
        )

    def call(self, features, graph, training=False):
        predict = super().call(features, training=training)
        propagate = self.ranker.propagate(graph, predict, graph_dropout=0.5 * training)
        return tf.nn.softmax(propagate, axis=1)

from timeit import default_timer as time


graph, features, labels = pg.load_feature_dataset("citeseer")

# this is how e cache the cross-origin resource
pre = pg.preprocessor(renormalize=True, assume_immutability=True, cors=True)
graph = pre(graph)


training, test = pg.split(list(range(len(graph))), 0.8, seed=5)
training, validation = pg.split(training, 1 - 0.2 / 0.8)
model = APPNP(features.shape[1], labels.shape[1])
with pg.Backend("tensorflow"):  # pygrank computations in tensorflow backend

    tic = time()
    pg.gnn_train(
        model,
        features,
        graph,
        labels,
        training,
        validation,
        epochs=300,
        optimizer=tf.optimizers.Adam(learning_rate=0.01),
        verbose=True,
        test=test,
    )
    print("Accuracy", pg.gnn_accuracy(labels, model(features, graph=graph), test))
    print("Time", time()-tic)
```

