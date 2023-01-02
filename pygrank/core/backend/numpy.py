import numpy as np
from numpy import abs, sum, exp, log, copy, repeat, min, max, dot, mean, diag, ones
from scipy.sparse import eye


def cast(x):
    return x


def backend_init():
    pass


def graph_dropout(M, _):
    return M


def separate_cols(x):
    return [x[:, col_num] for col_num in range(x.shape[1])]


def combine_cols(cols):
    return np.column_stack(cols)


def backend_name():
    return "numpy"


def scipy_sparse_to_backend(M):
    return M


def to_array(obj, copy_array=False):
    if isinstance(obj, np.ndarray):
        obj = np.asarray(obj)
        if copy_array:
            return np.copy(obj).squeeze()
        if len(obj.shape) > 1:
            return obj.squeeze()
        return obj
    if obj.__class__.__module__ == "tensorflow.python.framework.ops":
        return obj.numpy()
    if obj.__class__.__module__ == "torch":
        return obj.detach().numpy()
    return np.array(obj)


def to_primitive(obj):
    return np.array(obj, copy=False)


def is_array(obj):
    return isinstance(obj, list) or isinstance(obj, np.ndarray) or obj.__class__.__module__ == "tensorflow.python.framework.ops" or obj.__class__.__module__ == "torch"


def self_normalize(obj):
    np_sum = obj.__abs__().sum()
    if np_sum != 0:
        obj = obj / np_sum
    return obj


def conv(signal, M):
    return signal * M


def length(x):
    if isinstance(x, np.ndarray):
        if len(x.shape) > 1:
            return x.shape[0]*x.shape[1]
        return x.shape[0]
    return len(x)


def degrees(M):
    return np.asarray(sum(M, axis=1)).ravel()


def filter_out(x, exclude):
    return x[exclude == 0]


def epsilon():
    #return np.finfo(np.float32).eps
    return np.finfo(float).eps
