import tensorflow as tf
import numpy as np
from tensorflow import abs, reduce_sum as sum, exp, eye, identity as copy, reduce_min as min, reduce_max as max
import tensorflow.math.log as log


def backend_name():
    return "tensorflow"


def repeat(value, times):
    return tf.ones(shape=(times, 1), dtype=tf.float64)*value # default repeat creates an 1D tensor


def scipy_sparse_to_backend(M):
    coo = M.tocoo()
    return tf.SparseTensor([[coo.col[i], coo.row[i]] for i in range(len(coo.col))], tf.convert_to_tensor(coo.data, dtype=tf.float64), coo.shape)


def to_array(obj):
    if isinstance(obj, tf.Tensor) and (len(obj.shape)==1 or obj.shape[1] == 1):
        return obj
    return tf.convert_to_tensor([[v] for v in obj], dtype=tf.float64)


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
    return tf.reshape(tf.sparse.reduce_sum(M, axis=0), (-1,1))
