import networkx as nx
import numpy as np
import scipy

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


class MethodHasher:
    """ Used to hash methods."""

    def __init__(self, method, assume_immutability=True):
        self.assume_immutability = assume_immutability
        self._method = method
        self._stored = {}

    def __call__(self, *args, **kwargs):
        if self.assume_immutability:
            desc = str(args)+str(kwargs)
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