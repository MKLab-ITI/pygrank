import tensorflow as tf
import numpy as np
import warnings
from tensorflow import (
    abs,
    reduce_sum as sum,
    eye,
    exp,
    identity as copy,
    reduce_min as min,
    reduce_max as max,
    reduce_mean as mean,
    ones,
)

__pygrank_tf_config = {"mode": "dense"}


def cast(x):
    return tf.cast(x, dtype=tf.float32)


def backend_init(mode="dense"):
    __pygrank_tf_config["mode"] = mode
    # print('Enabling float32 in keras backend')
    tf.keras.backend.set_floatx("float32")


def backend_config():
    return __pygrank_tf_config


def log(x):
    return tf.math.log(x)


def graph_dropout(M, dropout):
    if dropout == 0:
        return M
    if isinstance(M, tf.SparseTensor):
        return tf.SparseTensor(M.indices, tf.nn.dropout(M.values, dropout), M.shape)
    return tf.nn.dropout(M, dropout)


def separate_cols(x):
    return [tf.gather(x, [col_num], axis=1) for col_num in range(x.shape[1])]


def combine_cols(cols):
    if len(cols[0].shape) < 2:
        cols = [tf.reshape(col, (-1, 1)) for col in cols]
    return tf.concat(cols, axis=1)


def backend_name():
    return "tensorflow"


def dot(x, y):
    return tf.reduce_sum(x * y)


def diag(diagonal, offset=0):
    return tf.linalg.diag(diagonal, k=offset)


def repeat(value, times):
    return (
        tf.ones(shape=(times,), dtype=tf.float32) * value
    )  # default repeat creates an 1D tensor


def scipy_sparse_to_backend(M):
    if __pygrank_tf_config["mode"] == "dense":
        try:
            return tf.convert_to_tensor(M.todense(), dtype=tf.float32)
        except MemoryError:
            warnings.warn(
                f'[pygrank.backend.tensorflow] Not enough memory to convert a scipy sparse matrix with shape {M.shape} to a numpy dense matrix before moving it to your device.\nWill create a tensorflow.SparseTensor instead.\nAdd the option mode="sparse" to the backend\'s initialization to hide this message.'
            )
        except tf.errors.ResourceExhaustedError:
            warnings.warn(
                f"[pygrank.backend.tensorflow] Not enough memory to move a numpy dense matrix with shape {M.shape} to the backend's device.\nWill create a tensorflow.SparseTensor instead.\nAdd the option mode=\"sparse\" to the backend's initialization to hide this message."
            )
    coo = M.tocoo()
    return tf.SparseTensor(
        [[coo.col[i], coo.row[i]] for i in range(len(coo.col))],
        tf.convert_to_tensor(coo.data, dtype=tf.float32),
        coo.shape,
    )


def to_array(obj, copy_array=False):
    if isinstance(obj, tf.Tensor):
        if len(obj.shape) != 1:
            obj = tf.reshape(obj, (-1,))
        if copy_array:
            return tf.identity(obj)
        return obj
    return tf.convert_to_tensor([v for v in obj], dtype=tf.float32)


def to_primitive(obj):
    return tf.convert_to_tensor(obj, dtype=tf.float32)


def is_array(obj):
    if isinstance(obj, list):
        return True
    return isinstance(obj, tf.Tensor) or isinstance(obj, tf.Variable)


def self_normalize(obj):
    np_sum = sum(abs(obj))
    return obj / np_sum if np_sum != 0 else obj


def conv(signal, M):
    if isinstance(M, tf.SparseTensor):
        return tf.reshape(
            tf.sparse.sparse_dense_matmul(M, tf.reshape(signal, (-1, 1))), (-1,)
        )
    return tf.reshape(M @ tf.reshape(signal, (-1, 1)), (-1,))


def length(x):
    if isinstance(x, tf.Tensor) or isinstance(x, tf.Variable):
        return x.shape[0]
    return len(x)


def degrees(M):
    if isinstance(M, tf.SparseTensor):
        return tf.reshape(tf.sparse.reduce_sum(M, axis=0), (-1,))
    return tf.reshape(tf.reduce_sum(M, axis=1), (-1,))  # TODO: investigate why different reduction axis is needed


def filter_out(x, exclude):
    return x[tf.reshape(tf.math.equal(exclude, 0), (-1,))]


def epsilon():
    return tf.keras.backend.epsilon()


def shape0(M) -> int:
    return M.shape[0]
