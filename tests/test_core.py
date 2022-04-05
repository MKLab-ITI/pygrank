import networkx as nx

import pygrank as pg
import pytest


def supported_backends():
    for backend in ["pytorch", "tensorflow", "torch_sparse", "numpy"]:
        pg.load_backend(backend)
        yield backend


def test_call_management():
    def test_func(x, y=0):
        return x+y
    assert pg.call(test_func, {"x":1, "y": 2, "z": 3}) == 3
    assert pg.call(test_func, {"y": 2, "z": 3}, [1]) == 3
    assert len(pg.remove_used_args(test_func, {"y": 2, "z": 3}, [1])) == 1
    with pytest.raises(Exception):
        pg.call(test_func, {"y": 2, "z": 3}, [1, 2])
    with pytest.raises(Exception):
        pg.PageRank(krylov_dims=5)


def test_primitive_conversion():
    for _ in supported_backends():
        assert pg.obj2id("str") == str(hash("str"))
        assert pg.sum(pg.to_array([1, 2, 3])) == 6
        assert pg.sum(pg.dot(pg.exp(pg.log(pg.to_array([4, 5]))), pg.to_array([2, 2]))) == 18
        primitive = pg.to_array([1, 2, 3])
        assert id(primitive) == id(pg.to_array(primitive, copy_array=False))
        assert id(primitive) != id(pg.to_array(primitive, copy_array=True))


def test_separate_and_combine():
    for _ in supported_backends():
        table = pg.to_primitive([[1, 2, 3], [4, 5, 6]])
        cols = pg.separate_cols(table)
        assert len(cols) == 3
        for col in cols:
            assert pg.length(col) == 2
        new_table = pg.combine_cols(cols)
        assert pg.sum(pg.abs(table-new_table)) == 0


def test_signal_init():
    for backend in supported_backends():
        with pytest.raises(Exception):
            pg.GraphSignal([1, 2, 3], [1, 2])
        signal = pg.GraphSignal(next(pg.load_datasets_graph(["graph9"])), {"A": 1, "B": 2})
        if backend != "tensorflow":
            del signal["A"]
            assert signal["A"] == 0
        assert signal["B"] == 2


def test_unimplemented_rank():
    with pytest.raises(Exception):
        pg.NodeRanking().rank(next(pg.load_datasets_graph(["graph9"])))


def test_backend_load():
    pg.load_backend("tensorflow")
    assert pg.backend_name() == "tensorflow"
    pg.load_backend("numpy")
    assert pg.backend_name() == "numpy"
    with pytest.raises(Exception):
        pg.load_backend("unknown")
    assert pg.backend_name() == "numpy"


def test_backend_with():
    for backend_name in ["pytorch", "tensorflow", "numpy", "torch_sparse"]:
        with pg.Backend(backend_name) as backend:
            assert pg.backend_name() == backend_name
            assert backend.backend_name() == backend_name
        assert pg.backend_name() == "numpy"


def test_signal_np_auto_conversion():
    import tensorflow as tf
    import numpy as np
    graph = nx.DiGraph([(1, 2), (2, 3)])
    signal = pg.to_signal(graph, tf.convert_to_tensor([1., 2., 3.]))
    assert isinstance(signal.np, np.ndarray)
    with pg.Backend("tensorflow"):
        assert pg.backend_name() == "tensorflow"
        assert not isinstance(signal.np, np.ndarray)
    assert pg.backend_name() == "numpy"
    assert isinstance(signal.np, np.ndarray)


def test_signal_direct_operations():
    graph = nx.DiGraph([(1, 2), (2, 3)])
    signal = pg.to_signal(graph, [1., 2., 3.])
    assert pg.sum(signal) == 6
    assert pg.sum(signal+1) == 9
    assert pg.sum(1+signal) == 9
    assert pg.sum(signal**2) == 14
    assert pg.sum(signal-[1, 2, 2]) == 1
    assert pg.sum(-1+signal) == 3
    assert pg.sum(signal / pg.to_signal(graph, [1., 2., 3.])) == 3
    assert pg.sum(3**signal) == 3+9+27
    signal **= 2
    assert pg.sum(signal) == 14
    signal.np = pg.to_signal(graph, [4, 4, 4])
    assert pg.sum(signal) == 12
    assert pg.sum(+signal) == 12
    assert pg.sum(-signal) == -12
    assert pg.sum(-signal/2) == -6
    assert pg.sum(-signal//2) == -6
    assert pg.sum(2/signal) == 1.5
    assert pg.sum(2//signal) == 0
    signal += 1
    assert pg.sum(signal) == 15
    signal -= 1
    assert pg.sum(signal) == 12
    signal /= 2
    assert pg.sum(signal) == 6
    signal //= 2
    assert pg.sum(signal) == 3
    signal *= 4
    assert pg.sum(signal) == 12
    with pytest.raises(Exception):
        signal+pg.to_signal(graph.copy(), [1., 2., 3.])
