# Setup 

Create a (virtual) environment with Python 3.9 or later
and install or upgrade to the latest version of `pygrank` with:

```bash
pip install --upgrade pygrank
```

## Creating graphs

When working in practical problems,
use the [networkx](https://networkx.org/) 
library to construct graphs
by adding edges between objects.
For example, create a graph
that `pygrank` can process with the
following pattern, which is used 
in most documentation examples.

```python
import networkx as nx

graph = nx.Graph(directed=False)  # undirected is also the default
graph.add_edge('A', 'B')
graph.add_edge('A', 'C')
```

Graphs like the above require a lot of memory to keep track of relations
between datam which can be an issue when processing large graphs.
On the other hand, 
we are typically interested in 
converting those graphs to sparse matrices of respective
backends. For this reason, we provide our own
trimmed down graph class that speeds up the `add_edge` method
and implements only a subset of 
of operations needed for node ranking algorithms. 
Create instances of this class like so:

```python
import pygrank as pg

graph = pg.Graph(directed=False)  # undirected is also the default
graph.add_edge('A', 'B')
graph.add_edge('A', 'C')
```

## Backends

Several popular computational backends are supported.
To avoid bloat of the main package,
these should be installed separately as needed.
Only the default`"numpy"` can be used out-of-the box.
Find instructions on how to install and enable the rest below.

!!! info
    First-time users can stick to the default backend and skip the rest of this section.
    However, setting up graph analysis on GPUs with other backends can be hundreds of times faster.

To switch between backends, either use the `load_backend(name)`
command or define an execution context that temporarily switches
to the specified backend and then reverts to the previous one. 
This is the recommended approach, as demonstrated below. 
Switching backends only affects how new operations are executed. Data types are automatically converted as needed, and caching optimizations are tied to the backend.


```python
import pygrank as pg

algorihtm = pg.PageRank()
with pg.Backend("tensorflow"):  # tensorflow needs to be installed
    scores = algorihtm(...)
    print(scores.np)  # a tensor
print(scores.np)  # an array now that we switched back
```

When importing `pygrank` a message appears indicating  that `"numpy"` is the default backend.
The same message points to a JSON configuration file stored under *home/.pygrank*,
alongside any automatically downloaded content. The configuration 
file specifies the default backend to be set upon the library's
first import, initialization parameters for that backend, and the option 
to silence the reminder message. These options can either be 
edited directly on the file or programmatically set with:

```python
pg.set_backend_preference(name, reminder=True, **init)  # essentially call pg.load_backend(name, **init) on pygrank's first import
```

The `init` dictionary holds parameters passed to backend initialization.
The configuration file's contents looks like this:

```json
{
  "backend": "numpy", 
  "reminder": "true",
  "init": {}
}
```

Below is a list of supported backends with installation instructions and comments.

### <span class="component">numpy</span>
<b class="parameters">About</b><br>This is the default backend and is enabled by default. Internally,
it employs `scipy` for sparse-dense matrix operations. All other backends rely on `scipy` sparse matrices
as an intermediate step when initializing their own sparse matrix types. This backend is
best suited to general-purpose numerical computations and
handling very large graphs with memory efficiency, but is not
the fastest option.
<br>
<b class="parameters">Links</b><br> [numpy](https://numpy.org/)<br>[scipy](https://scipy.org/)

### <span class="component">tensorflow</span>
<b class="parameters">About</b><br>Performs computations within the `tensorflow` execution environment.
The latter is an open-source platform for machine learning developed by the Google Brain team.
There 
are two modes in which this backend can be executed: `"dense"` (default) and `"sparse"`.
The mode may be provided as additional arguments to the backend loading call like this:

```python
import pygrank as pg
with pg.Backend("tensorflow", mode="dense", device="auto"):
    ... # code to run on pytorch here
```

In dense mode, the tensorflow backend attempts to store graphs in dense square
matrices that take full advantage of tensorflow's parallelization.
If there is not enough memory to allocate a sparse adjacency matrix,
the backend generates a sparse version and creates a warning.
The backend's initialization also accepts a device string or object to
which computations should be internally transferred. If provided, this needs to
be a tensorflow device name.
<br>
<b class="parameters">Installation</b><br> `pip install tensorflow[and-cuda]`<br>On Windows install WSL2 (Windows Subsystem for Linux) first.<br>
<b class="parameters">Links</b><br> [tensorflow](https://www.tensorflow.org/install)


### <span class="component">pytorch</span>
<b class="parameters">About</b><br>Performs computations within the `pytorch` execution environment.
The latter is an open-source platform for machine learning developed by Meta's AI Research lab.
Similarly to `"tensorflow"`, 
are two modes in which this backend can be executed: `"dense"` (default) and `"sparse"`.
The mode may be provided as additional arguments to the backend loading call like this:

```python
import pygrank as pg
with pg.Backend("pytorch", mode="dense", device="auto"):
    ... # code to run on pytorch
```

In dense mode, the pytorch backend attempts to store graphs in dense square
matrices that take full advantage of pytorch's device parallelization.
If there is not enough memory to allocate a sparse adjacency matrix,
the backend generates a sparse version and creates a warning.
The backend's initialization also accepts a device string or object to
which computations should be internally transferred. If provided, this needs to
be one among pytorch's available devices (typically `"cuda"` or `"cpu"`).
If not provided, the device will be the same as the one selected during the 
last time this backend was loaded. If this is the first time,
the device will be automatically selected to be `"cuda"`
if the latter is properly integrated, and `"cpu"` otherwise.
<br>
<b class="parameters">Installation</b><br> For full installation instructions visit pytorch's website in the links below.<br>
<b class="parameters">Links</b><br> [pytorch](https://pytorch.org/get-started/locally)

### <span class="component">torch_sparse</span>
<b class="parameters">About</b><br>Performs computations within the `pytorch` execution environment,
but contrary to the `"pytorch` backend uses the sparse computations of the `torch_sparse` library.
The latter is an open-source platform for machine learning developed by Meta's AI Research lab.
This backend always executes on sparse mode
and its initialization accepts a device string or object to
which computations should be internally transferred. This follows the
same conventions as `"pytorch"` to determine the employed device. For example,
use this backend like this:

```python
import pygrank as pg
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with pg.Backend("torch_sparse", device=device): 
    ...  # code to run on torch_sparse 
```

!!! info
    `"torch_sparse"` is near-identical to `"pytorch"`
    in sparse mode but is much faster in preprocessing adjacency matrices.

<b class="parameters">Installation</b><br> For full installation instructions visit pytorch's website in the links below.<br>
<b class="parameters">Links</b><br> [pytorch](https://pytorch.org/get-started/locally) <br>
[torch_sparse](https://github.com/rusty1s/pytorch_sparse)

### <span class="component">matvec</span>
<b class="parameters">About</b><br>Offers multithreaded implementations and memory reuse that are much faster that `"numpy"`
when processing extremely sparse graphs. It very fast when the number of edges is a small multiple of the number of nodes,
but is slower than other backends for dense graphs.
<br>
<b class="parameters">Installation</b><br> `pip install matvec`<br>
<b class="parameters">Links</b><br> [matvec](https://github.com/maniospas/matvec)


### <span class="component">dask</span>
<b class="parameters">About</b><br>Offers the distributed computational model of dask.distributed. 
Enables distributed computing and parallel processing, making it ideal for very large graphs that need 
to be processed in a distributed manner. 
This backend's instantiation accepts additional positional and a keyword argument `chunks=8` to denote
the number of chunks to which sparse matrices are split (the maximum number of engaged 
distributed works), and keyword arguments to pass to the dask client's constructor.
<br>
<b class="parameters">Installation</b><br> `pip install dask[distributed]`<br>
<b class="parameters">Links</b><br> [dask.distributed](https://distributed.dask.org/en/stable/)

### <span class="component">sparse_dot_mkl</span>
<b class="parameters">About</b><br>Running computations on parallelized scipy multiplications.
Provides speedups for sparse matrix multiplications by utilizing optimized MKL routines. 
Best suited when Intel's hardware and software stack are available.
<br>
<b class="parameters">Installation</b><br> `pip install sparse_dot_mkl` <br>
<b class="parameters">Links</b><br> [mkl](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html)

!!! info
    If you use Intel's Python distribution, `"sparse_dot_mkl"` is only marginally faster than `"numpy"`.