![pygrank](docs/pygrank.png)

Fast node ranking algorithms on large graphs.

**Author:** Emmanouil (Manios) Krasanakis
<br>**License:** Apache 2.0

![build](https://github.com/MKLab-ITI/pygrank/actions/workflows/tests.yml/badge.svg)
![coverage](coverage.svg)
[![Downloads](https://static.pepy.tech/personalized-badge/pygrank?period=total&units=international_system&left_color=black&right_color=orange&left_text=Downloads)](https://pepy.tech/project/pygrank)

# :hammer_and_wrench: Installation
`pygrank` requires Python 3.9 or later. Get the latest version per:

```bash
pip install --upgrade pygrank
```

Also install any of these optional dependencies to use the respective backend: `tensorflow`,`pytorch`,`torch_sparse`,`matvec`

# :link: Documentation

**https://pygrank.readthedocs.io**

# :brain: Overview

`pygrank` is a collection of node ranking algorithms 
and practices that support real-world conditions, 
such as large graphs and heterogeneous preprocessing 
and postprocessing requirements. Thus, it provides 
ready-to-use tools that simplify the deployment of 
theoretical advancements and testing of new algorithms.


# :thumbsup: Contributing
Feel free to contribute in any way, for example through the [issue tracker](https://github.com/MKLab-ITI/pygrank/issues) or by participating in [discussions]().
Please check out the [contribution guidelines](CONTRIBUTING.md) to bring modifications to the code base.
If so, make sure to **follow the pull checklist** described in the guidelines.
 
# :notebook: Citation
If `pygrank` has been useful in your research and you would like to cite it in a scientific publication, please refer to the following paper:
```
@article{krasanakis2022pygrank,
  author       = {Emmanouil Krasanakis, Symeon Papadopoulos, Ioannis Kompatsiaris, Andreas Symeonidis},
  title        = {pygrank: A Python Package for Graph Node Ranking},
  journal      = {SoftwareX},
  year         = 2022,
  month        = oct,
  doi          = {10.1016/j.softx.2022.101227},
  url          = {https://doi.org/10.1016/j.softx.2022.101227}
}
```
To publish research that makes use of provided implementations,
please cite their [relevant publications](docs/tips/citations.md).
