from pygrank.core import backend
from numpy.linalg import norm
import numpy as np
import warnings

def diags(vecs, offs):
    return np.add.reduce([np.diag(v,k) for v,k in zip(vecs, offs)])


def krylov_base(M, s, krylov_space_degree):
    warnings.warn("Krylov approximation is still under development")
    # TODO: throw exception for non-symmetric matrix
    s = backend.to_array(s)
    base = [s / backend.dot(s, s)**0.5]
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
    e1 = backend.repeat(0.0, krylov_space_degree)
    e1[0] = 1
    return (V @ filterH) @ e1