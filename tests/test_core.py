import pygrank as pg
import pytest


def supported_backends():
    for backend in ["pytorch", "tensorflow", "numpy"]:
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
