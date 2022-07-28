import warnings

import matvec as mv
import numpy as np
from matvec import abs, exp, log, dot, max, min, mean, repeat
from scipy.sparse import eye


# TODO: for full integration of this backend, add `matvec` to test_core.supported_backends


def sum(x):   # pragma: no cover
    if isinstance(x, mv.Vector) or isinstance(x, mv.Matrix):
        return mv.sum(x)
    return np.array(x)


def ones(x):   # pragma: no cover
    raise Exception("matvec does not implement ones")


def diag(x):   # pragma: no cover
    raise Exception("matvec does not implement diag")


def copy(x):   # pragma: no cover
    return x.copy()


def backend_init():   # pragma: no cover
    warnings.warn("Matvec is an experimental backend")


def graph_dropout(M, _):   # pragma: no cover
    return M


def separate_cols(x):   # pragma: no cover
    raise Exception("matvec does not support column separation")


def combine_cols(cols):   # pragma: no cover
    raise Exception("matvec does not support column combination")


def backend_name():   # pragma: no cover
    return "matvec"


def scipy_sparse_to_backend(M):   # pragma: no cover
    M = M.tocoo()
    return mv.Matrix(M.row, M.col, M.data, M.shape[0])


def to_array(obj, copy_array=False):   # pragma: no cover
    if isinstance(obj, mv.Vector):
        if copy_array:
            return copy(obj)
        return obj
    if obj.__class__.__module__ == "tensorflow.python.framework.ops":
        return mv.Vector(obj.numpy())
    if obj.__class__.__module__ == "torch":
        return mv.Vector(obj.detach().numpy())
    return mv.Vector(obj)


def to_primitive(obj):   # pragma: no cover
    if isinstance(obj, mv.Vector) or isinstance(obj, mv.Matrix):
        return obj
    return mv.to_vector(obj)


def is_array(obj):   # pragma: no cover
    return isinstance(obj, mv.Vector) or isinstance(obj, list) or isinstance(obj, np.ndarray) or obj.__class__.__module__ == "tensorflow.python.framework.ops" or obj.__class__.__module__ == "torch"


def self_normalize(obj):   # pragma: no cover
    np_sum = mv.sum(obj.__abs__())
    if np_sum != 0:
        obj.assign(obj / np_sum)
    return obj


def conv(signal, M):   # pragma: no cover
    return mv.multiply(M, signal)


def length(x):   # pragma: no cover
    return len(x)


def degrees(M):   # pragma: no cover
    return sum(M)


def filter_out(x, exclude):   # pragma: no cover
    return x[exclude == 0]


def epsilon():   # pragma: no cover
    return np.finfo(np.float64).eps
