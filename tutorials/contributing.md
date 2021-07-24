<center><h1>:thumbsup: Contributing</h1></center>

Feel free to provide any kind of code base contribution. This could include
implementing new publications, fixing, writting unit tests,
improving algorithms and extending the documentation.

You can also contribute through the [issue tracker].
Many thanks to all existing and future contributors for their participation.

# :hammer_and_wrench: Workflow
The typical workflow for `pygrank` contributions comprises the following steps:
1. **Fork** the master branch from the GitHub repository.
2. **Clone** the fork locally.
3. **Edit** the library.
4. **Commit** changes.
5. **Push** changes to the fork.
6. **Create a pull request** from the fork back to the original master branch.

You can use any (virtual) environment to edit the local clone,
such as conda or the one provided by PyCharm.
The environment should come with Python 3.6 or later installed.
Make sure that library dependencies 
`tqdm, sklearn, scipy, numpy, networkx`
are installed and upgraded to their latest versions.


# :white_check_mark: Pull Checklist
Before creating a pull request, make sure that your submission checks the following points:
1. There are docstring for new classes and methods adhering to [Google docstring](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
conventions.
2. Run `python docgenerator.py` to automatically add new docstrings.
3. All unit tests are passed with no errors, unless your purpose
is to introduce new unit tests that reveal bugs. Refrain from remodularizing
the code unless absolutely necessary (creating new packages is fine).
4. Unit tests provide 100% code coverage.
5. When implementing new or existing research (we are more than happy to accomodate this),
you are required to also update the library's [citations](!citations.md) and point to
that research from the respective class's docstring.
6. New code maintains CamelCase notation for classes and lower_case_with_underscores
for methods and variables.
7. New files are placed in what feels like appropriate packages.
7. **Optional.** Algorithms exhibit near-linear
(e.g. polylog-linear) running times and memory allocation to scale well to 
large graphs.

# :pencil2: Implementing New Node Ranking Algorithms
##### Which classes to subclass?
To create a new node ranking algorithm, you are required to subclass one of the
classes found in `pygrank.algorithms.abstract_filters`:
* `GraphFilter` identifies generic graph filters (is subclassed by the next two)
* `RecursiveGraphFilter` identifies graph filters that can be described with a recursive formula
* `ClosedFormGraphFilter` identifies graph filers that can be described in closed form

##### Where to write code?
New abstract classes (e.g. that define families of new algorithms) should be placed
in the same module as the above ones. New algorithms should be placed in modules
`pygrank.algorithms.[family]`, where *family* is either an existing
submodule or a new one. For new submodules, make sure to provide access to
their classes through `pygrank.algorithms.__init__.py`
(this is **important**, as it helps `docgenerator.py` to  automatically create
documentation for new algorithms).

##### How to structure constructors?
Constructors of graph filters should pass extra arguments to parent classes.
This ensures that new algorithms share the same breadth of customization
as parent classes. Only additional arguments not parsed by parent classes
(inherited arguments will be automatically added when `docgenerator.py`
is used to construct documentation). For example, the following snippet
presents a new algorithm:
 

```python
class NewAlgorithm(GraphFilter):
    def __init__(self, parameter=1, *args, **kwargs):
        """
        Instantiates the new algorithm.
        Args:
            parameter: Optional. The new algorithm's parameter. Default value is 1.
        """
        super().__init__(*args, **kwargs)
        self.parameter = parameter
    ...
```

Do not forget to follow the previously outlined pull checklist.

# :pencil2: Implementing New Evaluation Measures
TODO
