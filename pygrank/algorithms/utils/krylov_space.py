import numpy as np
from numpy import dot
from numpy.linalg import norm
import sklearn
import warnings

def diags(vecs, offs):
    return np.add.reduce([np.diag(v,k) for v,k in zip(vecs, offs)])


def krylov_base(M, s, krylov_space_degree):
    warnings.warn("Krylov approximation is still under development")
    try:
        sklearn.utils.validation.check_symmetric(M, tol=1e-10, raise_warning=False, raise_exception=True)
    except:
        raise Exception("Krylov approximation can only be performed for \"symmetric\" adjacency matrix normalization during preprocessing")
    base = [s / norm(s, 2)]
    base_norms = []
    alphas = []
    for j in range(0, krylov_space_degree):
        v = base[j]
        w = M*v
        a = dot(v, w)
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
    e1 = np.repeat(0.0, krylov_space_degree)
    e1[0] = 1
    return (V @ filterH) @ e1