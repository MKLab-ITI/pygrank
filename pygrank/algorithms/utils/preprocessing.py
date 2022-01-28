import networkx as nx
import numpy as np
import scipy
from pygrank import backend
import uuid


def to_sparse_matrix(G, normalization="auto", weight="weight", renormalize=False):
    """ Used to normalize a graph and produce a sparse matrix representation.

    Args:
        G: A networkx graph
        normalization: Optional. The type of normalization can be "col", "symmetric" or "auto" (default). The latter
            selects the type of normalization depending on whether the graph is directed or not respectively.
        weight: Optional. The weight attribute of the graph'personalization edges.
        renormalize: Optional. If True, the renormalization trick employed by graph neural networks to ensure iteration
            stability by shrinking the spectrum is applied. Default is False.
    """
    normalization = normalization.lower()
    if normalization == "auto":
        normalization = "col" if G.is_directed() else "symmetric"
    M = nx.to_scipy_sparse_matrix(G, weight=weight, dtype=float)
    if renormalize:
        M = M + scipy.sparse.eye(M.shape[0])*float(renormalize)
    if normalization == "col":
        S = np.array(M.sum(axis=1)).flatten()
        S[S != 0] = 1.0 / S[S != 0]
        Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
        M = Q * M
    elif normalization == "symmetric":
        S = np.array(np.sqrt(M.sum(axis=1))).flatten()
        S[S != 0] = 1.0 / S[S != 0]
        Qleft = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
        S = np.array(np.sqrt(M.sum(axis=0))).flatten()
        S[S != 0] = 1.0 / S[S != 0]
        Qright = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
        M = Qleft * M * Qright
    elif normalization != "none":
        raise Exception("Supported normalizations: none, col, symmetric, auto")
    return backend.scipy_sparse_to_backend(M)


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


def preprocessor(normalization="auto", assume_immutability=False, renormalize=False):
    """ Wrapper function that generates lambda expressions for the method to_sparse_matrix.

    Args:
        normalization: Normalization parameter for `to_sparse_matrix` (default is "auto").
        assume_immutability: If True, then the output is further wrapped through a MethodHasher to avoid redundant
            calls. Default is False, as graph immutability needs be explicitly assumed but cannot be guaranteed.
        renormalize: Graph renormalization parameter for `to_sparse_matrix` (default is False).
    """
    if assume_immutability:
        return MethodHasher(preprocessor(normalization, False, renormalize))
    return lambda G: to_sparse_matrix(G, normalization=normalization, renormalize=renormalize)
