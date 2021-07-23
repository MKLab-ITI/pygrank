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
1. Any new classes and methods are documented and adhering to [Google docstring](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
conventions.
2. All unit tests are passed with no errors, unless your purpose
is to introduce new unit tests that reveal existing bugs. Refrain from remodularizing
the code unless absolutely necessary (creating new packages is fine).
3. When implementing new or existing research (we are more than happy to accomodate this),
you are required to also update the library's [citations](!citations.md) and point to
that research from the respective class docstring.
4. New code should maintain CamelCase notation for classes and lower_case_with_underscores
for methods and variables.
5. New files are placed in what feels like appropriate packages.
6. **Optional.** Algorithm implementations should preferably exhibit near-linear
(e.g. polylog-linear) running times to scale well. 