import pytest
import pygrank as pg
from .test_core import supported_backends


def test_preprocessor_types():
    def test_graph():
        return next(pg.load_datasets_graph(["graph5"]))
    for _ in supported_backends():
        from random import random
        graph = test_graph()
        signal = pg.to_signal(graph, {v: random() for v in graph})
        laplacian = pg.preprocessor(normalization="laplacian")(graph)
        symmetric = pg.preprocessor(normalization="symmetric")(graph)
        assert pg.abs(pg.sum(pg.conv(signal, laplacian) + pg.conv(signal, symmetric) - signal)) <= pg.epsilon()


def test_preprocessor():
    def test_graph():
        return next(pg.load_datasets_graph(["graph5"]))

    for _ in supported_backends():
        graph = test_graph()
        with pytest.raises(Exception):
            pre = pg.preprocessor(normalization="unknown", assume_immutability=True)
            pre(graph)

        pre = pg.preprocessor(normalization="col", assume_immutability=False)
        graph = test_graph()
        res1 = pre(graph)
        res2 = pre(graph)
        assert id(res1) != id(res2)

        pre = pg.MethodHasher(pg.preprocessor, assume_immutability=True)
        graph = test_graph()
        res1 = pre(graph)
        pre.assume_immutability = False  # have the ability to switch immutability off midway
        res2 = pre(graph)
        assert id(res1) != id(res2)

        pre = pg.preprocessor(normalization="col", assume_immutability=True)
        graph = test_graph()
        res1 = pre(graph)
        pre.clear_hashed()
        res2 = pre(graph)
        assert id(res1) != id(res2)
