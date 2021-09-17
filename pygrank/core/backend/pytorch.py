import torch
import numpy as np
from torch import abs, sum, exp, eye, clone as copy, min, max, log


def backend_init():
    pass


def graph_dropout(M, dropout):
    if dropout == 0:
        return M
    # TODO: change based on future sparse matrix support: https://github.com/pytorch/pytorch/projects/24#card-59611437
    return torch.sparse_coo_tensor(M.indices, torch.dropout(M.values, dropout), M.shape).to_sparse_csr()


def separate_cols(x):
    return torch.split(x, dim=1)


def combine_cols(cols):
    return torch.cat(cols, dim=1)


def backend_name():
    return "tensorflow"


def dot(x, y):
    return torch.sum(x*y)


def repeat(value, times):
    return torch.ones(times)*value


def scipy_sparse_to_backend(M):
    coo = M.tocoo()
    return torch.sparse_coo_tensor(torch.LongTensor(np.vstack((coo.col, coo.row))), torch.FloatTensor(coo.data), coo.shape)


def to_array(obj, copy_array=False):
    if isinstance(obj, torch.Tensor) and (len(obj.shape) == 1 or obj.shape[1] == 1):
        if copy_array:
            return torch.clone(obj)
        return obj
    return torch.ravel(torch.FloatTensor(np.array([[v] for v in obj])))


def is_array(obj):
    if isinstance(obj, list):
        return True
    return isinstance(obj, torch.Tensor)


def self_normalize(obj):
    np_sum = sum(abs(obj))
    return obj/np_sum if np_sum != 0 else obj


def conv(signal, M):
    return torch.mv(M, signal)


def length(x):
    if isinstance(x, torch.Tensor):
        return x.shape[0]
    return len(x)


def degrees(M):
    # this sparse sum creates sparse matrices that need to be converted to dense to use in Hadamard products
    return torch.sparse.sum(M, dim=0).to_dense()


def epsilon():
    return torch.finfo(torch.float32).eps
