import numpy as np
from numpy import abs, sum, exp, log, copy, repeat, min, max, dot
from scipy.sparse import eye


def backend_name():
    return "numpy"


def scipy_sparse_to_backend(M):
    return M


def to_array(obj):
    if isinstance(obj, np.ndarray):
        return obj
    return np.array(obj)


def is_array(obj):
    return isinstance(obj, list) or isinstance(obj, np.ndarray)


def self_normalize(obj):
    np_sum = obj.__abs__().sum()
    if np_sum != 0:
        obj /= np_sum
    return obj


def conv(signal, M):
    return signal * M


def length(x):
    if isinstance(x, np.ndarray):
        return x.shape[0]
    return len(x)


def degrees(M):
    return np.asarray(sum(M, axis=1)).ravel()