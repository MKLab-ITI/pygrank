# Autotuning

Beyond the ability to compare node ranking algorithms,
we provide the ability to automatically tune node ranking 
algorithms or select the best ones with respect to optimizing a measure
based on the graph and personalization at hand. This process is abstracted
through a `pygrank.Tuner` base class, which wraps
any kind of node ranking algorithm. Ideally, this would wrap end-product
algorithms.

!!! info
    Tuners differ from benchmarks in that they select node ranking algorithms
    on-the-fly based on the graph signal input.

## Getting started

An exhaustive list of ready-to-use tuners can be found [here](../generated/tuners.md).
After initialization, these can run with the same pattern as other node ranking algorithms.
Tuner instances with default arguments use common base settings.
For example, the following code separates training and evaluation
data for a provided personalization signal and then uses a tuner that
by default creates a `GenericGraphFilter` instance with ten parameters.

```python
import pygrank as pg

_, graph, group = pg.load_one("eucore")
signal = pg.to_signal(graph, group)

train, test = pg.split(signal, training_samples=0.5)

scores_pagerank = pg.PageRank(max_iters=1000)(train)
scores_tuned = pg.ParameterTuner()(train)

measure = pg.AUC(test, exclude=train)
pg.benchmark_print_line("Pagerank", measure(scores_pagerank))
pg.benchmark_print_line("Tuned", measure(scores_tuned))
# Pagerank       	 .83
# Tuned          	 .91
```

Instead of repeating the whole optimization
process each time a tuner runs, you may
want to tune once and use the created node ranking
algorithm later. This can be achieved with the following pattern:

```python
algorithm_tuned = pg.ParameterTuner().tune(training)
scores_tuned = algorithm_tuned(training)
```

## Customization

Tune your algorithms by passing to the `ParameterTuner` 
a method (or lambda expression) that constructs them 
given a list of parameters. Also provide corresponding
upper and lower bounds for the parameters.
An example follows:

```python
def custom_algorithm(params): 
    assert len(params) == 1
    return pg.PageRank(alpha=params[0])

algorithm = pg.ParameterTuner(custom_algorithm, 
                                     max_vals=[0.99], 
                                     min_vals=[0.5],
                                     measure=pg.NDCG)
```


In the above snippet, we used the NDCG as the measure of choice for tuning.
If no measure is provided, AUC is the default. If the application calls
for it and you want to create a measure that is tied to a specific graph signal
with the `as_supervised_method` like below, set *fraction_of_training=1* for the tuner. This
forces the tuner to use the whole personalization to produce node ranks internally, 
since we perform the validation split a priori. 

```python
import pygrank as pg

_, graph, group = pg.load_one("eucore")
signal = pg.to_signal(graph, group)

train, test = pg.split(signal, training_samples=0.5)
train, valid = pg.split(train, training_samples=0.5)

tuner = pg.ParameterTuner(lambda params: pg.PageRank(alpha=params[0]),
                             max_vals=[0.99],
                             min_vals=[0.5],
                             fraction_of_training=1,
                             measure=pg.NDCG(valid, exclude=train+test).as_supervised_method())

scores_pagerank = pg.PageRank(max_iters=1000)(train)
scores_tuned = tuner(train)
```

## Optimizations

Graph convolutions are the most computationally-intensive operations
node ranking algorithms employ, as their running time scales linearly with the 
number of network edges (instead of nodes). However, when tuners
aim to optimize algorithms involving graph filters extending the
`ClosedFormGraphFilter` class, graph filtering is decomposed into 
weighted sums of naturally occurring
Krylov space base elements {*M<sup>n</sup>p*, *n=0,1,...*}.
To speed up computation time (by many times in some settings) `pygrank`
provides the ability to save the generation of this Krylov space base
so that future runs do *not* recompute it, effectively removing the need
to perform graph convolutions all but once for each personalization.

!!! info
    This speedup can be applied outside of tuners too;
    explicitly pass a graph signal object to node ranking algorithms.

To enable this behavior, a dictionary needs to be passed to closed form
graph filter constructors through an `optimization_dict` argument.
In most cases, tuners are responsible for delegating additional arguments
to default algorithms and this can be achieved with the following code.

```python
graph, personalization = ...
optimization_dict = dict()
tuner = pg.ParameterTuner(error_type="iters", 
                              num_parameters=20,
                              max_iters=20,
                              optimization_dict=optimization_dict)
scores = tuner(graph, personalization)
```

!!! warning
    Similarly to the `assume_immutability=True` option
    for preprocessors, the optimization dictionary requires that graphs signals are not altered in
    the interim, although it is possible to clear signal values.
    Furthermore, using optimization dictionaries multiplies (e.g. at least doubles)
    the amount of used memory, which the system may run out of for large graphs.
    To remove allocated memory, keep a reference to the dictionary and clear
    it afterwards with `optimization_dict.clear()`.

!!! info
    The default algorithms constructed by tuners (if none are provided) use
    *pygrank.SelfClearDict* instead of a normal dictionary. This clears other entries when
    a new personalization is inserted, therefore avoiding memory bloat.
