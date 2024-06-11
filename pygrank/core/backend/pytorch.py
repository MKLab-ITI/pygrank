import torch
import numpy as np
from torch import abs, clone as copy, log, exp
import warnings


__pygrank_torch_config = {"device": "auto", "mode": "dense"}


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
    return torch.diagflat(x, offset=offset).to(__pygrank_torch_config["device"])


def backend_init(mode="dense", device=None):
    __pygrank_torch_config["mode"] = mode
    if device is not None and device == "auto":
        if (
            not isinstance(__pygrank_torch_config["device"], str)
            or __pygrank_torch_config["device"] != "auto"
        ):
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        warnings.warn(
            f"[pygrank.backend.pytorch] Automatically detected device to run on {device}: {torch.get_device(device)}"
        )
    if device is not None and isinstance(device, str):
        device = torch.device(device)
    __pygrank_torch_config["device"] = device


def backend_config():
    return __pygrank_torch_config


def graph_dropout(M, dropout):
    if dropout == 0:
        return M
    if not M.is_sparse:
        return torch.nn.functional.dropout(M, dropout)
    # TODO: change based on future sparse matrix support: https://github.com/pytorch/pytorch/projects/24#card-59611437
    return torch.sparse_coo_tensor(
        M.indices(), torch.nn.functional.dropout(M.values(), dropout), M.shape
    )  # .coalesce()


def separate_cols(x):
    return torch.unbind(x, dim=1)


def combine_cols(cols):
    if len(cols[0].shape) < 2:
        cols = [torch.reshape(col, (-1, 1)) for col in cols]
    return torch.cat(cols, dim=1)


def backend_name():
    return "pytorch"


def dot(x, y):
    return torch.sum(x * y)


def ones(*args):
    return torch.ones(*args, device=__pygrank_torch_config["device"])


def eye(*args):
    return torch.eye(*args, device=__pygrank_torch_config["device"])


def repeat(value, times):
    return torch.ones(times, device=__pygrank_torch_config["device"]) * value


def scipy_sparse_to_backend(M):
    if __pygrank_torch_config["mode"] == "dense":
        try:
            return torch.FloatTensor(M.todense()).to(__pygrank_torch_config["device"])
        except MemoryError:
            warnings.warn(
                f"[pygrank.backend.pytorch] Not enough memory to convert a scipy sparse matrix with shape {M.shape} "
                f"to a numpy dense matrix before moving it to your device.\nWill create a torch.sparse_coo_tensor instead."
                f'\nAdd the option mode="sparse" to the backend\'s initialization to hide this message,'
                f"\nbut prefer switching to the torch_sparse backend for faster preprocessing."
            )

    coo = M.tocoo()
    return (
        torch.sparse_coo_tensor(
            torch.vstack((torch.LongTensor(coo.col).to(__pygrank_torch_config["device"]),
                          torch.LongTensor(coo.row).to(__pygrank_torch_config["device"]))),
            torch.FloatTensor(coo.data).to(__pygrank_torch_config["device"]),
            coo.shape,
        )
        .coalesce()  # THIS IS MANDATORY TO GET FAST MULTIPLICATIONS
    )


def to_array(obj, copy_array=False):
    if isinstance(obj, torch.Tensor):
        if len(obj.shape) == 1 or obj.shape[1] == 1:
            if copy_array:
                return torch.clone(obj).to(__pygrank_torch_config["device"])
            return obj.to(__pygrank_torch_config["device"])
        return torch.ravel(obj).to(__pygrank_torch_config["device"])
    if not isinstance(obj, np.ndarray):
        from pygrank.core.backend import to_numpy
        obj = to_numpy(obj)
    return torch.ravel(
        torch.FloatTensor(np.array(obj, dtype=np.float32))
    ).to(__pygrank_torch_config["device"])


def to_primitive(obj):
    if isinstance(obj, float):
        return torch.tensor(obj, dtype=torch.float32).to(
            __pygrank_torch_config["device"]
        )
    return torch.FloatTensor(obj).to(__pygrank_torch_config["device"])


def is_array(obj):
    if isinstance(obj, list):
        return True
    return isinstance(obj, torch.Tensor)


def self_normalize(obj):
    np_sum = sum(abs(obj))
    return obj / np_sum if np_sum != 0 else obj


def conv(signal, M):
    # if M.is_sparse:
    return torch.mv(M, signal)
    # return M@signal.reshape((-1,1))


def length(x):
    if isinstance(x, torch.Tensor):
        return x.shape[0]
    return len(x)


def degrees(M):
    # this sparse sum creates sparse matrices that need to be converted to dense to use in Hadamard products
    if M.is_sparse:
        return torch.sparse.sum(M, dim=0).to_dense()
    return torch.sum(M, dim=1)


def filter_out(x, exclude):
    return x[exclude == 0]


def epsilon():
    return torch.finfo(torch.float32).eps


def shape0(M) -> int:
    return M.shape[0]
