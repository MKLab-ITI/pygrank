import networkx as nx
import sklearn.preprocessing
import scipy
import numpy as np


class ConvergenceManager:
    def __init__(self, tol=1.E-6, error_type="mabs", max_iters=100, allow_reset=True):
        self.tol = tol
        self.error_type = error_type.lower()
        self.max_iters = max_iters
        self.allow_reset = allow_reset
        self.iteration = 0
        self.last_ranks = None

    def reset(self):
        if self.allow_reset:
            self.iteration = 0
            self.last_ranks = None

    def has_converged(self, new_ranks):
        self.iteration += 1
        if self.iteration>self.max_iters:
            raise Exception("Could not converge within", self.max_iters, "iterations")
        converged = False if self.last_ranks is None else self._has_converged(self.last_ranks, new_ranks)
        self.last_ranks = new_ranks
        return converged

    def _has_converged(self, prev_ranks, ranks):
        ranks = np.array(ranks)
        if self.error_type == "msqrt":
            return (scipy.square(ranks - prev_ranks).sum()/ranks.size)**0.5 < self.tol
        elif self.error_type == "mabs":
            return scipy.absolute(ranks - prev_ranks).sum()/ranks.size < self.tol
        elif self.error_type == "small_value":
            return (scipy.absolute(ranks).sum()/ranks.size) < self.tol
        else:
            raise Exception("Supported error types: msqrt, mabs")


def to_scipy_sparse_matrix(G, normalization="auto", weight="weight"):
    normalization = normalization.lower()
    M = nx.to_scipy_sparse_matrix(G, weight=weight, dtype=float)
    if normalization == "auto":
        normalization = "col" if G.is_directed() else "symmetric"
    if normalization == "col":
        M = sklearn.preprocessing.normalize(M, "l1", axis=1, copy=False)
    elif normalization == "symmetric":
        M = sklearn.preprocessing.normalize(M, "l2", axis=0, copy=False)
        M = sklearn.preprocessing.normalize(M, "l2", axis=1, copy=False)
    elif normalization != "none":
        raise Exception("Supported normalizations: none, col, symmetric, auto")
    return M


def assert_binary(ranks):
    for v in ranks.values():
        if v not in [0, 1]:
            raise Exception('Binary ranks required')