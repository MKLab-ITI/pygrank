import numpy as np
from numpy import abs, sum, exp, log, copy, repeat, min, max, dot, mean, diag, ones
from scipy.sparse import eye as _eye
import dask as dsk
import dask.distributed


__pygrank_dask_config = {"splits": 0, "client": None}


def cast(x):
    return x


def backend_init(*args, splits: int = 8, client=None, **kwargs):
    __pygrank_dask_config["splits"] = splits
    if __pygrank_dask_config["client"] is None:
        if client is None:
            client = dsk.distributed.Client(*args, **kwargs)
        __pygrank_dask_config["client"] = client
    elif client is not None:
        __pygrank_dask_config["client"] = client
    return __pygrank_dask_config["client"]


def backend_config():
    return __pygrank_dask_config


def graph_dropout(M, _):
    return M


def separate_cols(x):
    return [x[:, col_num] for col_num in range(x.shape[1])]


def combine_cols(cols):
    return np.column_stack(cols)


def backend_name():
    return "dask"


def eye(*args):
    # return scipy_sparse_to_backend(_eye(*args))  # this creates an error for lanczos methods
    return _eye(*args)


def scipy_sparse_to_backend(M):
    M = M.tocsc()
    rows = M.shape[0]
    splt = __pygrank_dask_config["splits"]
    split_size = rows // splt
    splits = []
    for i in range(splt):
        start_index = i * split_size
        if i == splt - 1:  # last split includes the remaining rows
            end_index = rows
        else:
            end_index = (i + 1) * split_size
        splits.append(M[:, start_index:end_index])
    # return splits
    return __pygrank_dask_config["client"].scatter(splits)


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
    return (
        isinstance(obj, list)
        or isinstance(obj, np.ndarray)
        or obj.__class__.__module__ == "tensorflow.python.framework.ops"
        or obj.__class__.__module__ == "torch"
    )


def self_normalize(obj):
    np_sum = obj.__abs__().sum()
    if np_sum != 0:
        obj = obj / np_sum
    return obj


"""
def conv(signal, M):
    results = []
    for split in M:
        result = signal @ split
        results.append(result)
    final_result = np.concatenate(results)
    return final_result
"""


def conv(signal, M_splits):
    def multiply_and_collect(signal, split):
        return signal @ split

    # Use Dask to parallelize the multiplication
    futures = [
        __pygrank_dask_config["client"].submit(multiply_and_collect, signal, split)
        for split in M_splits
    ]
    results = __pygrank_dask_config["client"].gather(futures)

    final_result = np.concatenate(results, axis=0)
    return final_result


def length(x):
    if isinstance(x, list) and len(x) > 0 and isinstance(x[0], dask.distributed.Future):
        return sum(block.get().shape[0] * block.get().shape[1] for block in x)
    if isinstance(x, np.ndarray):
        if len(x.shape) > 1:
            return x.shape[0] * x.shape[1]
        return x.shape[0]
    return len(x)


def degrees(M):
    def degs(block):
        return np.asarray(sum(block, axis=1)).ravel()

    futures = [__pygrank_dask_config["client"].submit(degs, block) for block in M]
    results = __pygrank_dask_config["client"].gather(futures)

    ret = 0
    for result in results:
        ret = result + ret
    return ret


def filter_out(x, exclude):
    return x[exclude == 0]


def epsilon():
    # return np.finfo(np.float32).eps
    return np.finfo(float).eps


def shape0(M) -> int:
    return M[0].get().shape[0]
