import networkx as nx
import scipy
import numpy as np
import time


class ConvergenceManager:
    """ Used to keep previous iteration and generally manage convergence of variables.

    Examples:
        >>> convergence = ConvergenceManager()
        >>> convergence.start()
        >>> var = None
        >>> while not convergence.has_converged(var):
        >>>     ...
        >>>     var = ...
    """

    def __init__(self, tol=1.E-6, error_type="mabs", max_iters=100, allow_restart=True):
        self.tol = tol
        self.error_type = error_type.lower()
        self.max_iters = max_iters
        self.min_iters = 0
        self.allow_restart = allow_restart
        self.iteration = 0
        self.last_ranks = None
        self._start_time = None
        self.elapsed_time = None

    def start(self):
        if self.allow_restart or self.last_ranks is None:
            self.iteration = 0
            self.last_ranks = None
            self._start_time = time.clock()
            self.elapsed_time = None
            self.min_iters = 0

    def force_next_iteration(self):
        self.min_iters = self.iteration+1

    def has_converged(self, new_ranks):
        self.iteration += 1
        if self.iteration<=self.min_iters:
            return True
        if self.iteration>self.max_iters:
            raise Exception("Could not converge within", self.max_iters, "iterations")
        if self._start_time is None:
            raise Exception("Need to start() the convergence manager")
        converged = False if self.last_ranks is None else self._has_converged(self.last_ranks, new_ranks)
        self.last_ranks = new_ranks
        self.elapsed_time = time.clock()-self._start_time
        return converged

    def _has_converged(self, prev_ranks, ranks):
        if self.error_type == "const":
            return True
        ranks = np.array(ranks)
        if self.error_type == "msqrt":
            return (scipy.square(ranks - prev_ranks).sum()/ranks.size)**0.5 < self.tol
        elif self.error_type == "mabs":
            return scipy.absolute(ranks - prev_ranks).sum()/ranks.size < self.tol
        elif self.error_type == "small_value":
            return scipy.absolute(ranks).sum()/ranks.size < self.tol
        else:
            raise Exception("Supported error types: msqrt, mabs, const")

    def __str__(self):
        return str(self.iteration)+" iterations ("+str(self.elapsed_time)+" sec)"


def to_scipy_sparse_matrix(G, normalization="auto", weight="weight"):
    """ Used to normalize a graph and produce a sparse matrix representation.

    Attributes:
        G: A networkx graph
        normalization: The type of normalization can be "col", "symmetric" or "auto" (default). The latter selects
             one of the previous normalization depending on whether the graph is directed or not respectively.
        weight: The weight attribute of the graph's edges.
    """
    normalization = normalization.lower()
    if normalization == "auto":
        normalization = "col" if G.is_directed() else "symmetric"
    M = nx.to_scipy_sparse_matrix(G, weight=weight, dtype=float)
    if normalization == "col":
        S = scipy.array(M.sum(axis=1)).flatten()
        S[S != 0] = 1.0 / S[S != 0]
        Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
        M = Q * M
    elif normalization == "symmetric":
        S = scipy.array(np.sqrt(M.sum(axis=1))).flatten()
        S[S != 0] = 1.0 / S[S != 0]
        Qleft = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
        S = scipy.array(np.sqrt(M.sum(axis=0))).flatten()
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
            return to_scipy_sparse_matrix(*args, **kwargs)


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