<center><h1>:hammer_and_wrench: Functional Interface</h1></center> 

This functional interface builds upon concepts presented in the
[documentation](documentation.md) to simplify code written with
the *pygrank* library.

## Table of Contents
1. [Table of Contents](#table-of-contents)
2. [Chaining Postprocessors](#chaining-postprocessors)
3. [Graph Signals in Chains](#graph-signals-in-chains)
4. [Customizing Base Algorithms](#customizing-base-algorithms)


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
following snippet to normalized a pagerank algorithm's
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
which is equivalent to addint `use_quotient=False` in the filter's 
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