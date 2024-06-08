# Setup 

Create a (virtual) environment with Python 3.9 or later
and install or upgrade to the latest version of `pygrank` with:

```
pip install --upgrade pygrank
```

## Backends

Several popular computational backends are supported.
To avoid installation bloat of the main `pygrank` package,
these should be installed separately as needed.
Only `"numpy"` can be used immediately out-of-the box as
the default. Find instructions on how to install 
and enable the rest below.

!!! info
    First-time users can stick to the default and skip the rest of this section.

To switch between backends you can either use the `load_backend(name)`
command or define an execution context that temporarily boots the
backend and afterwards reverts to the previous one. 
This is the recommended behavior and demonstrated below. 
Switching backends is smooth in that it only affects how new operations
are executed. Data types are automatically converted as needed,
and caching optimizations are tied to the backend.


```python
import pygrank as pg

algorihtm = pg.PageRank()
with pg.Backend("tensorflow"):  # tensorflow needs to be installed
    scores = algorihtm(...)
    print(scores.np)  # a tensor
print(scores.np)  # an array now that we switched back
```

When importing `pygrank` a message appears show that `"numpy"` is the default backend.
The same message points to a configuration file that is stored under *home/.pygrank*.
In addition to automatically downloaded content, there is a json configuration file
with the default backend that is to be set upon first import and the option to silence
the reminder message. The configuration looks like this and can be edited (or you can
programmatically edit it with `pg.set_backend_preference(name, reminder=True)`):

```json
{
  "backend": "numpy", 
  "reminder": "true"
}
```

Below is a list of supported backends with installation instructions and comments.

### <span class="component">numpy</span>
<b class="parameters">About</b><br>This is the default backend and is enabled by default. Internally,
it employs scipy for sparse-dense matrix operations. All other backends rely on scipy sparse matrices
as an intermediate step of creating their own sparse matrix types.
<br>
<b class="parameters">Links</b><br> [numpy](https://numpy.org/)<br>[scipy](https://scipy.org/)

### <span class="component">tensorflow</span>
<b class="parameters">About</b><br>Performs computations within the `tensorflow` execution environment.
<br>
<b class="parameters">Installation</b><br> `pip install tensorflow[and-cuda]`<br>On Windows install WSL2 (Windows Subsystem for Linux) first.<br>
<b class="parameters">Links</b><br> [tensorflow](https://www.tensorflow.org/install)


### <span class="component">pytorch</span>
<b class="parameters">About</b><br>Performs computations within the `pytorch` execution environment.
<br>
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
Backend instantiation accepts additional positional and a keyword argument `chunks=8` to denote
the number of chunks to which sparse matrices are split (the maximum number of engaged 
distributed works), and keyword arguments to pass to the instantiated dask client.
<br>
<b class="parameters">Installation</b><br> `pip install dask[distributed]`<br>
<b class="parameters">Links</b><br> [dask.distributed](https://distributed.dask.org/en/stable/)

### <span class="component">sparse_dot_mkl</span>
<b class="parameters">About</b><br>Running computations on parallelized scipy multiplications.
<br>
<b class="parameters">Installation</b><br> `pip install sparse_dot_mkl` <br>
<b class="parameters">Links</b><br> [mkl](https://www.intel.com/content/www/us/en/developer/tools/oneapi/distribution-for-python.html)

!!! info
    If you use Intel's Python distribution, this is only marginally faster than `"numpy"`.