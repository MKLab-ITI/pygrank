from pygrank.core.typing import BackendGraph, BackendPrimitive
from typing import Iterable, Optional, Tuple


def backend_name() -> str:  # pragma: no cover
    return "no backend loaded"


def backend_init():  # pragma: no cover
    pass


def graph_dropout(M: BackendGraph, dropout: BackendPrimitive) -> BackendPrimitive:  # pragma: no cover
    pass


def separate_cols(x: BackendPrimitive) -> BackendPrimitive:  # pragma: no cover
    pass


def combine_cols(cols: Iterable[BackendPrimitive]) -> BackendPrimitive:  # pragma: no cover
    pass


def abs(x: BackendPrimitive) -> BackendPrimitive:  # pragma: no cover
    pass


def sum(x: BackendPrimitive, axis: Optional[int] = None) -> BackendPrimitive:  # pragma: no cover
    pass


def mean(x: BackendPrimitive, axis: Optional[int] = None) -> BackendPrimitive:  #pragma: no cover
    pass


def min(x: BackendPrimitive, axis: Optional[int]=None) -> BackendPrimitive:  # pragma: no cover
    pass


def max(x: BackendPrimitive, axis: Optional[int]=None) -> BackendPrimitive:  # pragma: no cover
    pass


def exp(x: BackendPrimitive) -> BackendPrimitive:  # pragma: no cover
    pass


def log(x: BackendPrimitive) -> BackendPrimitive: # pragma: no cover
    pass


def ones(dims: Tuple[int, int]) -> BackendPrimitive:  # pragma: no cover
    pass


def eye(dims: int) -> BackendPrimitive:  # pragma: no cover
    pass


def diag(diagonal: BackendPrimitive, offset: int = 0) -> BackendPrimitive:  # pragma: no cover
    pass


def copy(x: BackendPrimitive) -> BackendPrimitive:  # pragma: no cover
    pass


def scipy_sparse_to_backend(M: BackendGraph) -> BackendGraph:  # pragma: no cover
    pass


def to_array(obj: object, copy_array: bool = False) -> BackendPrimitive:  # pragma: no cover
    pass


def to_primitive(obj: object) -> BackendPrimitive:  # pragma: no cover
    pass


def is_array(obj: object) -> bool:  # pragma: no cover
    pass


def repeat(value: float, times: int) -> BackendPrimitive:  # pragma: no cover
    pass


def self_normalize(obj: BackendPrimitive) -> BackendPrimitive:  # pragma: no cover
    pass


def conv(signal: BackendPrimitive, M: BackendGraph) -> BackendPrimitive:  # pragma: no cover
    pass


def length(x: BackendPrimitive) -> int:  # pragma: no cover
    pass


def degrees(M: BackendGraph) -> BackendPrimitive:  # pragma: no cover
    pass


def dot(x: BackendPrimitive, y: BackendPrimitive) -> BackendPrimitive:   # pragma: no cover
    pass


def filter_out(x: BackendPrimitive, exclude: BackendPrimitive) -> BackendPrimitive:   # pragma: no cover
    pass


def epsilon() -> float:   # pragma: no cover
    pass
