import tensorflow as tf
import numpy as np
from tensorflow import abs, reduce_sum as sum, exp, eye, identity as copy, repeat


def scipy_sparse_to_backend(M):
    coo = M.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def to_array(obj):
    if isinstance(obj, tf.Tensor) and obj.shape[1] == 1:
        return obj
    return tf.convert_to_tensor([[v] for v in obj])


def is_array(obj):
    if isinstance(obj, list):
        return True
    return isinstance(obj, tf.Tensor)


def self_normalize(obj):
    np_sum = sum(abs(obj))
    return obj/np_sum if np_sum != 0 else obj


def conv(signal, M):
    return tf.sparse.sparse_dense_matmul(M, signal)


def length(x):
    if isinstance(x, tf.Tensor):
        return x.shape[0]
    return len(x)


def degrees(M):
    return tf.experimental.numpy.ravel(sum(M, axis=1))