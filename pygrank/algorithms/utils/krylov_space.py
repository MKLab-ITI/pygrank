from pygrank.core import backend
from pygrank import measures
from numpy.linalg import norm
import numpy as np
import warnings


def diags(vecs, offs):
    return np.add.reduce([np.diag(v,k) for v,k in zip(vecs, offs)])


def krylov_base(M, personalization, krylov_space_degree):
    warnings.warn("Krylov approximation is still under development")
    # TODO: throw exception for non-symmetric matrix
    personalization = backend.to_array(personalization)
    base = [personalization / backend.dot(personalization, personalization) ** 0.5]
    base_norms = []
    alphas = []
    for j in range(0, krylov_space_degree):
        v = base[j]
        w = backend.conv(M, v)
        a = backend.dot(v, w)
        alphas.append(a)
        next_w = w - a*v
        if j > 0:
            next_w -= base[j - 1] * base_norms[j - 1]
        next_w_norm = norm(next_w, 2)
        base_norms.append(next_w_norm)
        if j != krylov_space_degree - 1:
            base.append(next_w / next_w_norm)
    H = diags([alphas, base_norms[1:], base_norms[1:]], [0, -1, 1])
    V = np.column_stack(base)
    return V, H


def krylov2original(V, filterH, krylov_space_degree):
    if isinstance(V, int) or isinstance(V, float):
        V = np.ones((krylov_space_degree, krylov_space_degree))*V
    ret = (V @ filterH)
    return np.asarray(ret[:, 0]).squeeze()  # TODO: find why a matrix ends up here sometimes


def krylov_error_bound(V, H, M, personalization, measure=measures.Mabs, max_powers=1):
    personalization = personalization / backend.dot(personalization, personalization) ** 0.5
    krylov_dims = V.shape[1]
    krylov_result = backend.eye(int(krylov_dims))
    errors = list()
    for power in range(max_powers+1):
        errors.append(measure(personalization)(krylov2original(V, krylov_result, int(krylov_dims))))
        if power < max_powers:
            krylov_result = krylov_result @ H
            personalization = backend.conv(M, personalization)
    return max(errors)


def arnoldi_iteration(A, b, n: int):
    """Computes a basis of the (n + 1)-Krylov subspace of A: the space
    spanned by {b, Ab, ..., A^n b}.

    Source: https://en.wikipedia.org/wiki/Arnoldi_iteration

    Arguments:
      A: m × m array
      b: initial vector (length m)
      n: dimension of Krylov subspace, must be >= 1

    Returns:
      Q: m x (n + 1) array, the columns are an orthonormal basis of the
        Krylov subspace.
      h: (n + 1) x n array, A on basis Q. It is upper Hessenberg.
    """
    eps = 1e-12
    h = np.zeros((n + 1, n))
    Q = np.zeros((A.shape[0], n + 1))
    # Normalize the input vector
    Q[:, 0] = b / np.linalg.norm(b, 2)  # Use it as the first Krylov vector
    for k in range(1, n):
        v = backend.conv(A, Q[:, k - 1])  # Generate a new candidate vector
        for j in range(k):  # Subtract the projections on previous vectors
            h[j, k - 1] = np.dot(Q[:, j].T, v)
            v = v - h[j, k - 1] * Q[:, j]
        h[k, k - 1] = np.linalg.norm(v, 2)
        if h[k, k - 1] > eps:  # Add the produced vector to the list, unless
            Q[:, k] = v / h[k, k - 1]
        else:  # If that happens, stop iterating.
            return Q, h
    return Q, h
