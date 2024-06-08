# Functional Interface

This functional interface simplifies code written with *pygrank*.


## Chaining Postprocessors

The core methodology of building node ranking algorithms is by
selecting a starting base algorithm (e.g. a graph filter) and
passing these through various postprocessors to define more
complicated schemas. Some postprocessors iterate over their
base algorithms, so a full chain of wrapped algorithms is
first defined.

For example, to apply the sweep procedure on personalized
pagerank and normalize the final outputs by dividing with
their sum, the object-oriented interface of the library can 
define a new algorithm per:

```python
import pygrank as pg

algorithm = pg.Normalize("sum", pg.Sweep(pg.PageRank()))
```

This scheme can get rather complicated, so the library also
implements the `>>` operator to annotate transfer of algorithms.
Simply put, the above code can be written as:

```python
import pygrank as pg

algorithm = pg.PageRank() >> pg.Sweep() >> pg.Normalize("sum")
```

This indicates that we start from the first algorithm and then
wrap around it all subsequent postprocessors. All
postprocessor constructors can in principle be defined without 
a base algorithm, so the chain assigns those.

## Chaining Personalization Preprocessing

The same interface can be used interchangeably to assign values
to the `personalization_transform` argument of graph filters, 
which applies a node ranking algorithm to personalization graph signal
inputs before the main algorithm (although theoretically different,
this ends up using filters as ype of postprocessor).

For example, the following snippet applies node
ranking with neighborhood inflation on inputs and runs
the SALSA algorithm (pagerank with "salsa" graph normalization)
on the first outputted scores.

```python
import pygrank as pg

wtf_stochastic = pg.PageRank() >> pg.SeedOversampling("neighbors") >> pg.PageRank(normalization="salsa")
```

## Graph Signals in Chains

There are two ways to enter graph signals in chains: 
a) by chaining algorithm calls with graph signal inputs,
and b) by chaining a call to final node ranking algorithms
with signals. Of course, there is always the third option of simply 
defining algorithms and calling them for appropriate data.
In fact, this is the only option for fairness-aware algorithms,
but here we focus on chain notation capabilities.

Of the two options, the first is much more intuitive and allows
signals to simply be parsed by algorithms. For example,
given a graph and a list of seed nodes, you can write the 
following snippet to normalize a pagerank algorithm's
output:

```python
import pygrank as pg

graph, seeds = ...
ranks = pg.to_signal(graph, seeds) >> pg.PageRank() >> pg.Normalize()
```

What you need to remember is that chaining a signal into an algorithm
via the `>>` operator creates a new signal to be passed on to
future calls. Some -but not all- postprocessors 
just happen to support calling with no base ranking algorithm, in
which case the base algorithm is assumed to be a tautology. However,
iterative postprocessors are not able to follow this convention.
For example, substituting normalization with the
sweep procedure in the last snippet does not work and parenthesis
need to be added for the algorithm definition, somewhat defeating 
the whole purpose of chaining.

In cases where it is not possible to employ the above notation
but you still want to write only one expression, node ranking
algorithms implement the `|` operator to determine graph signal
inputs. For example, you can write:

```python
import pygrank as pg

graph, seeds = ...
ranks = pg.PageRank() >> pg.Sweep() >> pg.Normalize("sum") | pg.to_signal(graph, seeds)
```

Of course, it is always possible to write `signal >> algorithm`
instead of `algorithm(signal)` after algorithms have been defined.


## Customizing Base Algorithms

To simplify graph filter chains, we final provide the ability to
define convergence managers, preprocessors and the quotient strategy
of graph filters with an operator instead of parts of the constructor.
These are automatically integrated if they reside on the right hand of
graph filters with the `+` operator. For example, the following applies
a postprocessor to the pagerank filter and also sets the quotient
strategy (the output signal transform after each step) to a tautology,
which is equivalent to adding `use_quotient=False` in the filter's 
constructor:

```python
import pygrank as pg

graph, seeds = ...
pre = pg.preprocessor(normalization="symmetric")
algorithm = pg.PageRank() + pre + pg.Tautology() >> pg.Sweep() >> pg.Normalize("sum")
ranks = algorithm(graph, seeds)
```

Notice that `+` has higher priority than `>>`, which means that 
filters are adjusted before applying the chain and no parenthesis is needed.

# Algorithm Combinations
In addition to other functional operations, it is also possible to
combine node ranking algorithms. Simple transformations, such as 
multiplications of all ranking scores,
can be performed with postprocessors. But there are two operations
that are explicitly defined. First, algorithm output addition `&` applies
the `+` operation between graph signal outputs. This symbol is chosen
to obtain lower priority than `<<` so that the latter is performed
first. Second, algorithm output negation `~` applies a minus sign to output
signals. 

Together, these symbols enable the operation pattern `alg1 & ~alg2` that
induces differences between node ranking algorithms. This is particularly
important when creating `pygrank.GenericGraphFilter` instances with 
a zero at some parameters but numerical tolerance convergence criteria 
(which would stop at the first zero parameter) instead of criteria that 
stop at a fixed number of iterations.

For example, thanks to algorithm linearity, 
`pg.GenericGraphFilter([0, 0, 1])` to extract
two-hop friend-of-friends [scellato2011exploiting]
can instead be written as the difference between two filters with
`tol=None` (that stops at exactly zero numerical deviation between
consecutive graph signal propagation):

```python
import pygrank as pg

two_hop = pg.GenericGraphFilter([1, 1, 1], tol=None) & ~pg.GenericGraphFilter([1, 1], tol=None)
```