import torch
import numpy as np
from torch import abs, exp, clone as copy, log
import torch_sparse
import warnings


__pygrank_torch_sparse_config = {"device": "auto"}


class TorchSparseGraphData:
    def __init__(self, index, values, shape):
        self.index = index
        self.values = values
        self.shape = shape

    def to(self, device):
        return TorchSparseGraphData(
            self.index.to(device), self.values.to(device), self.shape
        )


def ones(*args):
    return torch.ones(*args, device=__pygrank_torch_sparse_config["device"])


def eye(*args):
    return torch.eye(*args, device=__pygrank_torch_sparse_config["device"])


def cast(x):
    return x.float()


def sum(x, axis=None):
    return torch.sum(x) if axis is None else torch.sum(x, dim=axis)


def max(x, axis=None):
    return torch.max(x) if axis is None else torch.max(x, dim=axis)


def min(x, axis=None):
    return torch.min(x) if axis is None else torch.min(x, dim=axis)


def mean(x, axis=None):
    return torch.mean(x) if axis is None else torch.mean(x, dim=axis)


def diag(x, offset=0):
    return torch.diagflat(x, offset=offset).to(__pygrank_torch_sparse_config["device"])


def backend_init(device="auto"):
    if device is not None and device == "auto":
        if (
            not isinstance(__pygrank_torch_sparse_config["device"], str)
            or __pygrank_torch_sparse_config["device"] != "auto"
        ):
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        warnings.warn(
            f"[pygrank.backend.torch_sparse] Automatically detected device to run on {device}: {torch.device(device)}"
        )
    if device is not None and isinstance(device, str):
        device = torch.device(device)
    __pygrank_torch_sparse_config["device"] = device


def backend_config():
    return __pygrank_torch_sparse_config


def graph_dropout(M, dropout):
    if dropout == 0:
        return M
    # TODO: change based on future sparse matrix support: https://github.com/pytorch/pytorch/projects/24#card-59611437
    index, values = torch_sparse.coalesce(
        M.index, torch.nn.functional.dropout(M.values, dropout), M.shape[0], M.shape[1]
    )
    return TorchSparseGraphData(index, values, M.shape)


def separate_cols(x):
    return torch.unbind(x, dim=1)


def combine_cols(cols):
    if len(cols[0].shape) < 2:
        cols = [torch.reshape(col, (-1, 1)) for col in cols]
    return torch.cat(cols, dim=1)


def backend_name():
    return "torch_sparse"


def dot(x, y):
    return torch.sum(x * y)


def repeat(value, times):
    return torch.ones(times) * value


def scipy_sparse_to_backend(M):
    coo = M.tocoo()
    index, values = torch_sparse.coalesce(
        torch.LongTensor(np.vstack((coo.col, coo.row))).to(
            __pygrank_torch_sparse_config["device"]
        ),
        torch.FloatTensor(coo.data).to(__pygrank_torch_sparse_config["device"]),
        coo.shape[0],
        coo.shape[1],
    )
    return TorchSparseGraphData(index, values, coo.shape)


def to_array(obj, copy_array=False):
    if isinstance(obj, torch.Tensor):
        if len(obj.shape) == 1 or obj.shape[1] == 1:
            if copy_array:
                return torch.clone(obj).to(__pygrank_torch_sparse_config["device"])
            return obj.to(__pygrank_torch_sparse_config["device"])
        return torch.ravel(obj).to(__pygrank_torch_sparse_config["device"])
    return torch.ravel(torch.FloatTensor(np.array([v for v in obj]))).to(
        __pygrank_torch_sparse_config["device"]
    )


def to_primitive(obj):
    if isinstance(obj, float):
        return torch.tensor(obj, dtype=torch.float32).to(
            __pygrank_torch_sparse_config["device"]
        )
    return torch.tensor(obj, dtype=torch.float32).to(__pygrank_torch_sparse_config["device"])


def is_array(obj):
    if isinstance(obj, list):
        return True
    return isinstance(obj, torch.Tensor)


def self_normalize(obj):
    np_sum = sum(abs(obj))
    return obj / np_sum if np_sum != 0 else obj


def conv(signal, M):
    signal = torch.reshape(signal, (-1, 1))
    return torch.ravel(
        torch_sparse.spmm(M.index, M.values, M.shape[0], M.shape[1], signal)
    )


def length(x):
    if isinstance(x, torch.Tensor):
        return x.shape[0]
    return len(x)


def degrees(M):
    signal = torch.reshape(
        torch.ones(M.shape[0], device=__pygrank_torch_sparse_config["device"]), (-1, 1)
    )
    index, values = torch_sparse.transpose(M.index, M.values, M.shape[0], M.shape[1])
    return torch.ravel(torch_sparse.spmm(index, values, M.shape[1], M.shape[0], signal))


def filter_out(x, exclude):
    return x[exclude == 0]


def epsilon():
    return torch.finfo(torch.float32).eps


def shape0(M) -> int:
    return M.shape[0]
