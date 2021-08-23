import tensorflow as tf
import numpy as np
from tensorflow import abs, reduce_sum as sum, exp, eye, identity as copy, reduce_min as min, reduce_max as max


def backend_init():
    # print('Enabling float64 in keras backend')
    tf.keras.backend.set_floatx('float64')


def graph_dropout(M, dropout):
    if dropout == 0:
        return M
    return tf.SparseTensor(M.indices, tf.nn.dropout(M.values, dropout), M.shape)


def separate_cols(x):
    return [tf.gather(x, [col_num], axis=1) for col_num in range(x.shape[1])]


def combine_cols(cols):
    return tf.concat(cols, axis=1)


def backend_name():
    return "tensorflow"


def log(x):
    return tf.math.log(x)


def dot(x, y):
    return tf.reduce_sum(x*y)


def repeat(value, times):
    return tf.ones(shape=(times, 1), dtype=tf.float64)*value # default repeat creates an 1D tensor


def scipy_sparse_to_backend(M):
    coo = M.tocoo()
    return tf.SparseTensor([[coo.col[i], coo.row[i]] for i in range(len(coo.col))], tf.convert_to_tensor(coo.data, dtype=tf.float64), coo.shape)


def to_array(obj, copy_array=False):
    if isinstance(obj, tf.Tensor) and (len(obj.shape)==1 or obj.shape[1] == 1):
        if copy_array:
            return tf.identity(obj)
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
