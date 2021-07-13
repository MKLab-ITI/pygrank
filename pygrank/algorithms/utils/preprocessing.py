import networkx as nx
import numpy as np
import scipy


def to_numpy_idx(nodes, queries):
    queries = set(queries)
    return [i for i, v in enumerate(nodes) if v in queries]


def to_numpy(nodes, node2values, normalization=True, autocomplete=True):
    if isinstance(node2values, np.ndarray):
        if nodes is not None and node2values.size != len(nodes):
            raise Exception("A preconverted numpy vector with different than the desired size is used", node2values.size, 'vs', len(nodes))
        if normalization:
            return node2values / node2values.sum()
        return node2values
    if not autocomplete:
        nodes = [n for n in nodes if n in node2values]
    vector = np.repeat(1.0, len(nodes)) if node2values is None else np.array([node2values.get(n, 0) for n in nodes], dtype=float)
    if normalization:
        if vector.sum() == 0:
            raise Exception("The personalization vector should contain at least one non-zero entity")
        vector = vector / vector.sum()
    return vector


def to_dict(nodes, ranks):
    if not isinstance(ranks, np.ndarray):
        return ranks
    return dict(zip([v for v in nodes], map(float, ranks)))


def to_scipy_sparse_matrix(G, normalization="auto", weight="weight"):
    """ Used to normalize a graph and produce a sparse matrix representation.

    Attributes:
        G: A networkx graph
        normalization: The type of normalization can be "col", "symmetric" or "auto" (default). The latter selects
             one of the previous normalization depending on whether the graph is directed or not respectively.
        weight: The weight attribute of the graph's edges.
        sensitive: The sensitivity attribute of the graph's nodes.
    """
    normalization = normalization.lower()
    if normalization == "auto":
        normalization = "col" if G.is_directed() else "symmetric"
    M = nx.to_scipy_sparse_matrix(G, weight=weight, dtype=float)
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
    return M


def assert_binary(ranks):
    """ Assert that ranks.values() are only 0 or 1 ."""
    for v in ranks.values():
        if v not in [0, 1]:
            raise Exception('Binary ranks required')


def _idfier(*args, **kwargs):
    return "["+",".join(str(id(arg)) for arg in args)+"]"+"{"+",".join(v+":"+str(id(kwargs[v])) for v in kwargs)+"}"


class MethodHasher:
    """ Used to hash methods."""

    def __init__(self, method, assume_immutability=True):
        self.assume_immutability = assume_immutability
        self._method = method
        self._stored = dict()

    def clear_hashed(self):
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


def preprocessor(normalization="auto", assume_immutability=False):
    """ Wrapper function that generates lambda expressions for the method to_scipy_sparse_matrix.

    Attributes:
        normalization: Normalization parameter for to_scipy_sparse_matrix (default is "auto").
        assume_immutability: If True, then the output is further wrapped through a MethodHasher to avoid redundant
            calls. Default is False, as graph immutability needs be explicitly assumed but cannot be guaranteed.
    """
    if assume_immutability:
        return MethodHasher(preprocessor(normalization, False))
    return lambda G: to_scipy_sparse_matrix(G, normalization=normalization)


def vectorize(normalize_vectors=True, autocomplete=True, assume_immutability=False):
    #if assume_immutability:
    #    return MethodHasher(vectorize(normalize_vectors, autocomplete, False))
    return lambda G, dictionary: to_numpy(G, dictionary, normalization=normalize_vectors, autocomplete=autocomplete)