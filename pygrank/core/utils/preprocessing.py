import networkx as nx
import numpy as np
import scipy
from pygrank.core import backend
from pygrank.fastgraph import fastgraph
import uuid


def eigdegree(M):
    """
    Calculates the entropy-preserving eigenvalue degree to be used in matrix normalization.
    Args:
        M: the adjacency matrix
    """
    v = backend.repeat(1., M.shape[0])
    from pygrank.algorithms import ConvergenceManager
    convergence = ConvergenceManager(tol=backend.epsilon(), max_iters=1000)
    convergence.start()
    eig = 0
    v = v / backend.dot(v, v)
    while not convergence.has_converged(eig):
        v = backend.conv(v, M)
        v = backend.safe_div(v, backend.dot(v, v)**0.5)
        eig = backend.dot(backend.conv(v, M), v)
    return eig/(v*v)


def to_sparse_matrix(G,
                     normalization="auto",
                     weight="weight",
                     renormalize=False,
                     reduction=backend.degrees,
                     transform_adjacency = lambda x: x,
                     cors=False):
    """ Used to normalize a graph and produce a sparse matrix representation.

    Args:
        G: A networkx or fastgraph graph. If an object with a "shape" attribute is provided (which means that it
            is backend matrix) then it is directly returned.
        normalization: Optional. The type of normalization can be "none", "col", "symmetric", "laplacian", "both",
            or "auto" (default). The last one selects the type of normalization between "col" and "symmetric",
            depending on whether the graph is directed or not respectively. Alternatively, this could be a callable,
            in which case it transforms a scipy sparse adjacency matrix to produce a normalized copy.
        weight: Optional. The weight attribute (default is "weight") of *networkx* graph edges. This is ignored when
            *fastgraph* graphs are parsed, as these are unweighted.
        renormalize: Optional. If True, the renormalization trick (self-loops) of graph neural networks is applied to
            ensure iteration stability by shrinking the graph's spectrum. Default is False. Can provide anything that
            can be cast to a float to regularize the renormalization.
        reduction: Optional. Controls how degrees are calculated from a callable (e.g. `pygrank.eigdegree`
            for entropy-preserving transition matrices [li2011link]). Default is `pygrank.degrees`.
        cors: Optional.<details><summary>Cross-origin resource (shared between backends). Default is false.</summary>
            If True, it enriches backend primitives
            holding the outcome of graph preprocessing with additional private metadata that enable their
            usage as base graphs when passing through other postprocessors in other backends.
            This is not required when constructing GraphSignal instances with
            the pattern `pygrank.to_signal(M, personalization_data)` where `M = pygrank.preprocessor(cors=True)(graph)`
            but is mandarotry when the two commands are called in different backends. Note that *cors* objects are not
            normalized again with other strategies in other preprocessors and compliance is not currently enforced.
            There is **significant speedup** in using *cors* when frequently switching between backends for the
            same graphs. Furthermore, after defining such instances, they can be used in place of base graphs.
            If False (default), a lot of memory is saved by not keeping pointers to all versions of adjacency matrices
            among backends that use them. Enabling *cors* and then visiting up to two backends out of which one is
            "numpy", does not affect the maximum memory consumption by code processing one graph.
            </details>
    """
    if hasattr(G, "__pygrank_preprocessed"):
        if backend.backend_name() in G.__pygrank_preprocessed:
            return G.__pygrank_preprocessed[backend.backend_name()]  # this is basically caching, but it's pretty safe for just passing adjacency matrices around
        ret = backend.scipy_sparse_to_backend(G.__pygrank_preprocessed["numpy"])
        if cors:
            ret.__pygrank_preprocessed = G.__pygrank_preprocessed
            ret.__pygrank_preprocessed[backend.backend_name()] = ret
        else:
            ret.__pygrank_preprocessed = {backend.backend_name(): ret}
        ret._pygrank_node2id = G._pygrank_node2id
        return ret
    with backend.Backend("numpy"):
        normalization = normalization.lower() if isinstance(normalization, str) else normalization
        if normalization == "auto":
            normalization = "col" if G.is_directed() else "symmetric"
        M = G.to_scipy_sparse_array() if isinstance(G, fastgraph.Graph) else nx.to_scipy_sparse_matrix(G, weight=weight, dtype=float)
        renormalize = float(renormalize)
        left_reduction = reduction #(lambda x: backend.degrees(x)) if reduction == "sum" else reduction
        right_reduction = lambda x: left_reduction(x.T)
        if renormalize != 0:
            M = M + scipy.sparse.eye(M.shape[0])*renormalize
        if normalization == "col":
            S = np.array(left_reduction(M)).flatten()
            S[S != 0] = 1.0 / S[S != 0]
            Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
            M = Q * M
        elif normalization == "laplacian":
            S = np.array(np.sqrt(left_reduction(M))).flatten()
            S[S != 0] = 1.0 / S[S != 0]
            Qleft = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
            S = np.array(np.sqrt(right_reduction(M))).flatten()
            S[S != 0] = 1.0 / S[S != 0]
            Qright = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
            M = Qleft * M * Qright
            M = -M + scipy.sparse.eye(M.shape[0])
        elif normalization == "both":
            S = np.array(left_reduction(M)).flatten()
            S[S != 0] = 1.0 / S[S != 0]
            Qleft = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
            S = np.array(right_reduction(M)).flatten()
            S[S != 0] = 1.0 / S[S != 0]
            Qright = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
            M = Qleft * M * Qright
        elif normalization == "symmetric":
            S = np.array(np.sqrt(left_reduction(M))).flatten()
            S[S != 0] = 1.0 / S[S != 0]
            Qleft = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
            S = np.array(np.sqrt(right_reduction(M))).flatten()
            S[S != 0] = 1.0 / S[S != 0]
            Qright = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
            M = Qleft * M * Qright
        elif callable(normalization):
            M = normalization(M)
        elif normalization != "none":
            raise Exception("Supported normalizations: none, col, symmetric, both, laplacian, auto")
    M = transform_adjacency(M)
    ret = M if backend.backend_name() == "numpy" or backend.backend_name() == "sparse_dot_mkl" else backend.scipy_sparse_to_backend(M)
    ret._pygrank_node2id = {v: i for i, v in enumerate(G)}
    if cors:
        ret.__pygrank_preprocessed = {backend.backend_name(): ret, "numpy": M}
        M.__pygrank_preprocessed = ret.__pygrank_preprocessed
    else:
        ret.__pygrank_preprocessed = {backend.backend_name(): ret}
    return ret


def assert_binary(ranks):
    """ Assert that ranks.values() are only 0 or 1.

        Args:
            ranks: A dict-like object (e.g. a GraphSignal) to check for violations.
    """
    for v in ranks.values():
        if v not in [0, 1]:
            raise Exception('Binary ranks required', v)


def obj2id(obj):
    if isinstance(obj, object) and not isinstance(obj, str):
        if not hasattr(obj, "uuid"):
            obj.uuid = uuid.uuid1()
        return str(obj.uuid)
    return str(hash(obj))


def _idfier(*args, **kwargs):
    """
    Converts args and kwargs into a hashable array of object ids.
    """
    return "[" +",".join(obj2id(arg) for arg in args) + "]" + "{" + ",".join(v + ":" + obj2id(kwarg) for v, kwarg in kwargs.items()) + "}" + backend.backend_name()


class MethodHasher:
    """ Used to hash method runs, so that rerunning them with the same object inputs would directly output
    the outcome of previous computations.

    Example:
        >>> from pygrank.algorithms.utils.preprocessing import MethodHasher
        >>> def method(x, deg=2):
        ...     print("Computing with params", x, deg)
        ...     return x**deg
        >>> hashed_method = MethodHasher(method)
        >>> print(hashed_method(2))
        Computing with params 2 2
        4
        >>> print(hashed_method(2, 3))
        Computing with params 2 3
        8
        >>> print(hashed_method(2))
        4
    """

    def __init__(self, method, assume_immutability=True):
        """
        Instantiates the method hasher for a given method.

        Args:
            method:
            assume_immutability: Optional. If True (default) then the hasher will produce the
                same outputs as previous runs of the same objects. If False, the instant is see-through
                and the method is run anew each time.
        """
        self.assume_immutability = assume_immutability
        self._method = method
        self._stored = dict()

    def clear_hashed(self):
        """
        Clears all hashed data.
        """
        self._stored = dict()

    def __call__(self, *args, **kwargs):
        if self.assume_immutability:
            desc = _idfier(*args, **kwargs)
            if desc in self._stored:
                return self._stored[desc]
            value = self._method(*args, **kwargs)
            self._stored[desc] = value
            return value
        else:
            return self._method(*args, **kwargs)


def preprocessor(normalization: str = "auto",
                 assume_immutability: bool = False,
                 weight: str = "weight",
                 renormalize: bool = False,
                 reduction=backend.degrees,
                 transform_adjacency=lambda x: x,
                 cors: bool = False):
    """ Wrapper function that generates lambda expressions for the method to_sparse_matrix.

    Args:
        normalization: Optional. The type of normalization can be "none", "col", "symmetric", "laplacian", "salsa",
            or "auto" (default). The last one selects the type of normalization between "col" and "symmetric",
            depending on whether the graph is directed or not respectively. Alternatively, this could be a callable,
            in which case it transforms a scipy sparse adjacency matrix to produce a normalized copy.
        weight: Optional. The weight attribute (default is "weight") of *networkx* graph edges. This is ignored when
            *fastgraph* graphs are parsed, as these are unweighted.
        assume_immutability: Optional. If True, the output of preprocessing further wrapped
            through a MethodHasher to avoid redundant calls. In this case, consider creating one
            `pygrank.preprocessor` and passing it to all algorithms running on the same graphs.
            Default is False, as graph immutability needs to be explicitly assumed but cannot be guaranteed.
        renormalize: Optional. If True, the renormalization trick (self-loops) of graph neural networks is applied to
            ensure iteration stability by shrinking the graph's spectrum. Default is False. Can provide anything that
            can be cast to a float to regularize the renormalization.
        reduction: Optional. Controls how degrees are calculated from a callable (e.g. `pygrank.eigdegree`
            for entropy-preserving transition matrices [li2011link]). Default is `pygrank.degrees`.
        cors: Optional.Cross-origin resource (shared between backends). Default is False.
            <details>
            If True, it enriches backend primitives
            holding the outcome of graph preprocessing with additional private metadata that enable their
            usage as base graphs when passing through other postprocessors in other backends.
            This is not required when constructing GraphSignal instances with
            the pattern `pygrank.to_signal(M, personalization_data)` where `M = pygrank.preprocessor(cors=True)(graph)`
            but is mandarotry when the two commands are called in different backends. Note that *cors* objects are not
            normalized again with other strategies in other preprocessors and compliance is not currently enforced.
            There may be speedups by using *cors* when frequently switching between backends for the
            same graphs. Usage is demonstrated in
            [GNN examples](/examples/publications/krasanakis2022pygrank/4.%20Autotune%20in%20APPNP.py) .
            If False (default), a lot of memory is saved by not keeping pointers to all versions of adjacency matrices
            among backends in which it is run. Overall, prefer keeping this behavior switched off. Enabling
            *cors* and then visiting up to two backends out of which one is "numpy", does not affect the maximum
            memory consumption by code processing one graph.
            </details>
    """
    if assume_immutability:
        ret = MethodHasher(preprocessor(assume_immutability=False,
                                        normalization=normalization, weight=weight, renormalize=renormalize,
                                        reduction=reduction, cors=cors, transform_adjacency=transform_adjacency))
        ret.__name__ = "preprocess"
        return ret

    def preprocess(G):
        return to_sparse_matrix(G, normalization=normalization, weight=weight, renormalize=renormalize,
                                reduction=reduction, cors=cors, transform_adjacency=transform_adjacency)

    return preprocess
