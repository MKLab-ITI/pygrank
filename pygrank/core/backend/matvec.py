import warnings

import matvec as mv
import numpy as np
from matvec import dot, max, min, mean, repeat
from scipy.sparse import eye


def cast(x):
    return x


def exp(x):
    if isinstance(x, mv.Vector):
        return mv.exp(x)
    return np.exp(np.array(x))


def log(x):
    if isinstance(x, mv.Vector):
        return mv.log(x)
    return np.log(np.array(x))


def abs(x):
    if isinstance(x, mv.Vector):
        return mv.abs(x)
    return np.sum(np.array(x))


def sum(x):
    if isinstance(x, mv.Vector) or isinstance(x, mv.Matrix):
        return mv.sum(x)
    return np.sum(np.array(x))


def ones(shape):
    return np.ones(shape)


def diag(x, offset=0):
    return np.diag(x, offset)


def copy(x):
    return x.copy()


def backend_init():
    warnings.warn("Matvec is an experimental backend")


def graph_dropout(M, _):
    return M


def separate_cols(x):
    return [mv.Vector(x[:, col]) for col in range(x.shape[1])]
    #raise Exception("matvec does not support column separation")


def combine_cols(cols):
    return np.column_stack([col.np() for col in cols])
    #raise Exception("matvec does not support column combination")


def backend_name():
    return "matvec"


def scipy_sparse_to_backend(M):
    M = M.tocoo()
    return mv.Matrix(M.row, M.col, M.data, M.shape[0])


def to_array(obj, copy_array=False):
    if isinstance(obj, np.matrix):
        return mv.Vector(np.ravel(np.asarray(obj)))
    if isinstance(obj, np.ndarray):
        return mv.Vector(obj.ravel())
    if isinstance(obj, mv.Vector):
        if copy_array:
            return copy(obj)
        return obj
    if obj.__class__.__module__ == "tensorflow.python.framework.ops":
        return mv.Vector(obj.numpy())
    if obj.__class__.__module__ == "torch":
        return mv.Vector(obj.detach().numpy())
    return mv.Vector(obj)


def to_primitive(obj):
    if isinstance(obj, mv.Vector) or isinstance(obj, mv.Matrix):
        return obj
    if isinstance(obj, np.ndarray) and len(obj.shape) > 1:
        return obj
    if isinstance(obj, list) and isinstance(obj[0], list):
        return np.array(obj)
    if isinstance(obj, float):
        return obj
    return mv.to_vector(obj)


def is_array(obj):
    return isinstance(obj, mv.Vector) or isinstance(obj, list) or isinstance(obj, np.ndarray) or obj.__class__.__module__ == "tensorflow.python.framework.ops" or obj.__class__.__module__ == "torch"


def self_normalize(obj):
    np_sum = mv.sum(obj.__abs__())
    if np_sum != 0:
        obj.assign(obj / np_sum)
    return obj


def conv(signal, M):
    return signal*M


def length(x):
    return len(x)


def degrees(M):
    return mv.sum(M, axis=0)


def filter_out(x, exclude):
    return x[exclude == 0]


def epsilon():
    return np.finfo(np.float64).eps*4
