import numpy as np
from numpy import abs, sum, exp, copy, repeat
from scipy.sparse import eye


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