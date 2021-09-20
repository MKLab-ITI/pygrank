<center><h1>:thumbsup: Contributing</h1></center>


Feel free to provide any kind of code base contribution. This could include
implementing new publications, fixing, writing unit tests,
improving algorithms and extending the documentation.

You can also contribute through the [issue tracker](https://github.com/MKLab-ITI/pygrank/issues).
Pull requests that address unassigned issues in the tracker are welcome.
Many thanks to all existing and future contributors for their participation.

# :hammer_and_wrench: Workflow
The typical workflow for `pygrank` contributions comprises the following steps:
1. **Fork** the master branch from the GitHub repository.
2. **Clone** the fork locally (recommended: also copy the *pre-commit* file to *.git/hooks*).
3. **Edit** the library.
4. **Commit** changes.
5. **Push** changes to the fork.
6. **Create a pull request** from the fork back to the original master branch.

You can use any (virtual) environment to edit the local clone,
such as conda or the one provided by PyCharm.
The environment should come with Python 3.6 or later installed.
Make sure that both base library dependencies 
`networkx, numpy, scipy, sklearn, wget`, as well as`tensorflow` and `torch` (to support
unit testing for the respective pipelines)
and `coverage` (to support code coverage measurements)
are installed and upgraded to their latest versions.

# :hammer_and_wrench: Architecture
`pygrank` adheres to a hierarchical architecture to manage inter-module dependencies,
which new code should maintain for import statements to work.
For example, do not design evaluation measures that depend on algorithms.
Rather, such components should be delegated to some of the other modules.
For reference, we re-iterate here the project's architecture. For more details,
please refer to the [documentation](documentation/documentation.md).

![architecture](documentation/architecture.png)

We ask that, when contributing new code, you try to import methods and 
classes through the highest-level 
architectural component they belong to that does not conflict with the code.
For example, to design a new filter you need import utility methods
from `pygrank.algorithms.utils`, since a higher-level import would create
self-recursions by trying to import all sub-modules. On the other hand,
in the same module you can safely import classes from `pygrank.measures`.


# :white_check_mark: Pull Checklist
Before creating a pull request, make sure that your submission checks the following points:
1. Class and method docstrings should adhere to [Google's docstring conventions](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
Additionally, code examples should be prefaced by a line starting with the word
`Example ` and ending in `:` and their code lines start with `>>>`.
2. When implementing new or existing research (we are more than happy to accomodate this),
you are required to also update the library's [citations](documentation/citations.md).
3. New code should maintain *CamelCase* notation for classes and 
*lower_case_with_underscores* for methods and variables.
4. New files should be placed in appropriate modules and new methods and classes
can be accessed from the top level.
5. Module dependencies should comply on the above-described architecture.
6. **[optional]** Algorithms should exhibit near-linear
(e.g. polylog-linear) running times and memory allocation with respect to
the number of edges, to scale well to large graphs.
7. **[pre-commit]** Run `python docgenerator.py` to add new classes to the documentation.
This can be automated for commits to the master branch
by copying the `pre-commit` script file to the local folder `.git/hooks`.
8. **[github actions]** Pass all unit tests with no errors, unless your purpose
is to introduce new unit tests that reveal existing bugs in the code base.
Refrain from remodularizing the code unless absolutely necessary
(creating new packages is fine, but backwards compatibility of import statements
is mandatory).
9. **[github actions]** Unit tests should provide 100% code coverage.


# :pencil2: Implementing New Node Ranking Algorithms
##### Which classes to subclass?
To create a new node ranking algorithm, you are required to subclass one of the
classes found in `pygrank.algorithms.filters.abstract_filters`:
* `GraphFilter` identifies generic graph filters (is subclassed by the next two)
* `RecursiveGraphFilter` identifies graph filters that can be described with a recursive formula
* `ClosedFormGraphFilter` identifies graph filers that can be described in closed form

Please extend this documentation if a new family of node ranking algorithms is implemented.

##### Where to write code?
* New abstract graph filter classes (e.g. that define families of new algorithms) should be placed
in the module `pygrank.algorithms.filters.abstract_filters`.
* New graph filters should be placed in modules
`pygrank.algorithms.filters.[family]`, where *family* is either an existing
submodule or a new one.
* For new filter families, make sure to provide access to
their classes through `pygrank.algorithms.filters.__init__.py`
(this is **important**, as it helps `docgenerator.py` automatically create
documentation for new algorithms).

##### Which method(s) to override?
Depending on which class you subclass, you need to override and implement a
different method; more general `GraphFilter` classes need to implement at
least a step `_step(M, personalization, ranks, *args, **kwargs)`
method of that class with the correct arguments
that implements iterative convolutions through the methods provided by the
backend  (these can be imported from `pygrank.backend`). You can add any
arguments and keyword arguments corresponding to additional graph signals,
but take care to put all algortithm-describing parameters in the constructor.
Note that `personalization` and `ranks` in this method are **graph signals**
but the method is expected to return the output of backend calculations, 
which will be directly assigned to `ranks.np`. The other two abstract subclasses
partially implement this method to simplify definition of new filters.

For `RecursiveGraphFilter` subclasses, you only need to implement the method
`_formula(self, M, personalization, ranks, *args, **kwargs)`, which describes
an iterative formula describing the filter. Contrary to above, for this method
`personalization` and `ranks` are **backend primitives**. Similarly to before,
it should also return a backend primitive.

For `ClosedFormGraphFilter` subclasses, you only need to implement the method
`_coefficient(self, previous_coefficient)` which calculates the next coefficient
*a<sub>n</sub>* of the closed form graph filter
*a<sub>0</sub>+a<sub>1</sub>M+a<sub>2</sub>M<sup>2</sup>+...*
given that the previous coefficient is *a<sub>n-1</sub>*, where the latter
is inputted as *None* when *a<sub>0</sub>* is being calculated. It may be
easier to use `n = self.convergence.iteration-1` to explicitly calculate the 
returned value.


##### How to structure constructors?
Constructors of graph filters should pass extra arguments to parent classes.
This ensures that new algorithms share the same breadth of customization
as parent classes. Only additional arguments not parsed by parent classes
need to be documented
(inherited arguments will be automatically added when `docgenerator.py`
is used to construct documentation). Try to parameterize constructors
as much as possible, to ensure that researchers can easily try different
variations.

As an example, the following snippet
introduces a new algorithm that performs a non-recursive
implementation of personalized PageRank based on the recursion
*a<sub>0</sub> = 1-alpha*, *a<sub>n</sub>=alpha a<sub>n-1</sub>* for *n>0*:
 

```python
class NewAlgorithm(ClosedFormGraphFilter):
    def __init__(self, alpha=0.85, **kwargs):
        """
        Instantiates the new algorithm.
        Args:
            alpha: Optional. The new algorithm'personalization parameter. Default value is 0.85.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
    
    def _coefficient(self, previous_coefficient):
        if previous_coefficient is None:
            return 1-self.alpha
        return previous_coefficient*self.alpha
    ...
```

Do not forget to follow the previously outlined pull checklist.

# :pencil2: Implementing New Postprocessors
#### Which class to subclass?
Postprocessors need to subclass the abstract
`pygrank.algorithms.postprocess.Postprocessor`
class.

#### Which method to override?
Postprocessors need to subclass the method `_transform(self, ranks)`
where `ranks` is a graph signal -typically the outcome of some other
node ranking algorithm (e.g. a graph filter). Implementations of this
method can return any kind of data convertible to graph signals,
such as dictionaries or backend primitives. When possible, use the latter
(i.e. manipulate `ranks.np` with backend operations and return the result)
to ensure faster computations through the graph filter pipeline, for example
when iterative postprocessors are applied afterwards.

# :pencil2: Implementing New Measures
TODO


# :pencil2: Implementing New Tuners
#### Which class to subclass?
Tuners need to subclass the abstract
`pygrank.algorithms.autotune.tuning.Tuner`
class.


### Which method to override?
Tuners need to override the method `_tune(self, graph, personalization, *args, **kwargs)`
whose arguments are all passed. This method should return a tuple `tuned_algorithm, personalization`,
where the tuned algorithm is generated by the tuner (ideally through some scheme that is passed to
the constructor) and the personalization signal should either be the original signal or a new one
obtained through a seamless backend pipeline from the original, so that it is backpropagate-able.
`tuned_algorithm(personalization)` will be used to procure the tuner's outcome.
At worst, return a `Tautology()` and the desired tuning outcome for the given personalization argument,
but try to avoid this if possible. Do note, that this could be necessary if Arnoldi or Lanczos
decompositions are supported.

### What to place in tuner constructors?
Ideally, new tuners should be able to obtain a preferred backend in which to perform tuning.
Refer to the implementation of existing tuners as a guideline of how to switch between backends.
Switching to different backends should ideally support backpropagation on the original signal;
thus, tuners should construct optimal node ranking algorithms, which are backpropagate-able.

Tuners should support as many types of algorithms as possible and thus parameterize-able 
generic methods to construct such algorithms could be added to the tuner constructors.

Finally, if Krylov space alternative are possible (e.g. Arnoldi decomposition), add an argument
`krylov_dims=None` that indicates either that those methods are not employed (None) or the 
number of krylov space dimensions.