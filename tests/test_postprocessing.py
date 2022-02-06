import pygrank as pg
import pytest
from .test_core import supported_backends


def test_abstract_postprocessor():
    graph = next(pg.load_datasets_graph(["graph5"]))
    with pytest.raises(Exception):
        p = pg.Postprocessor(pg.PageRank())
        p.rank(graph)


def test_tautology():
    graph = next(pg.load_datasets_graph(["bigraph"]))
    r = pg.PageRank().rank(graph)
    tr = pg.Tautology(pg.PageRank()).rank(graph)
    rt = pg.Tautology().transform(r)
    for u in graph:
        assert r[u] == rt[u]
        assert r[u] == tr[u]
    u = pg.Tautology().rank(graph)
    assert float(sum(u.np)) == len(graph)


def test_seed_oversampling_arguments():
    _, graph, group = next(pg.load_datasets_one_community(["graph9"]))
    with pytest.raises(Exception):
        pg.SeedOversampling(pg.PageRank(), 'unknown').rank(graph, {"A": 1})
    with pytest.raises(Exception):
        pg.SeedOversampling(pg.PageRank()).rank(graph, {"A": 0.1, "B": 1})
    with pytest.raises(Exception):
        pg.BoostedSeedOversampling(pg.PageRank(), 'unknown').rank(graph, {"A": 1})
    with pytest.raises(Exception):
        pg.BoostedSeedOversampling(pg.PageRank(), 'naive', oversample_from_iteration='unknown').rank(graph, {"B": 1})


def test_seed_oversampling():
    _, graph, group = next(pg.load_datasets_one_community(["graph9"]))
    for _ in supported_backends():
        training, evaluation = pg.split(list(group), training_samples=2)
        training, evaluation = pg.to_signal(graph, {v: 1 for v in training}), pg.to_signal(graph, {v: 1 for v in evaluation})
        for measure in [pg.NDCG, pg.AUC]:
            ranks = pg.PageRank(0.9, max_iters=1000).rank(graph, training)
            base_result = measure(evaluation, training).evaluate(ranks)
            ranks = pg.SeedOversampling(pg.PageRank(0.9, max_iters=1000)).rank(graph, training)
            so_result = measure(evaluation, training).evaluate(ranks)
            bso_result = measure(evaluation, training).evaluate(
                pg.BoostedSeedOversampling(pg.PageRank(0.9, max_iters=1000)).rank(graph, training))
            assert float(base_result) <= float(so_result)
            assert float(so_result) <= float(bso_result)
        pg.SeedOversampling(pg.PageRank(0.99, max_iters=1000), "top").rank(graph, training)
        pg.SeedOversampling(pg.PageRank(0.99, max_iters=1000), "neighbors").rank(graph, training)
        pg.BoostedSeedOversampling(pg.PageRank(max_iters=1000), 'naive', oversample_from_iteration='original').rank(graph, {"A": 1})


def test_norm_maintain():
    # TODO: investigate that 2.5*epsilon is truly something to be expected
    graph = next(pg.load_datasets_graph(["graph5"]))
    for _ in supported_backends():
        prior = pg.to_signal(graph, {"A": 2})
        posterior = pg.MabsMaintain(pg.Normalize(pg.PageRank(), "range")).rank(prior)
        assert abs(pg.sum(pg.abs(posterior.np)) - 2) < 2.5*pg.epsilon()


def test_normalize():
    import networkx as nx
    graph = next(pg.load_datasets_graph(["graph5"]))
    for _ in supported_backends():
        assert float(pg.sum(pg.Normalize("range").transform(pg.to_signal(nx.Graph([("A", "B")]), [2, 2])).np)) == 4
        r = pg.Normalize(pg.PageRank(), "range").rank(graph)
        assert pg.min(r.np) == 0
        assert pg.max(r.np) == 1
        r = pg.Normalize(pg.PageRank(), "sum").rank(graph)
        assert abs(pg.sum(r.np) - 1) < pg.epsilon()
        with pytest.raises(Exception):
            pg.Normalize(pg.PageRank(), "unknown").rank(graph)


def test_transform():
    import math
    graph = next(pg.load_datasets_graph(["graph5"]))
    for _ in supported_backends():
        r1 = pg.Normalize(pg.PageRank(), "sum").rank(graph)
        r2 = pg.Transformer(pg.PageRank(), lambda x: x/pg.sum(x)).rank(graph)
        assert pg.Mabs(r1)(r2) < pg.epsilon()
        r1 = pg.Transformer(math.exp).transform(pg.PageRank()(graph))
        r2 = pg.Transformer(pg.PageRank(), pg.exp).rank(graph)
        assert pg.Mabs(r1)(r2) < pg.epsilon()


def test_ordinals():
    graph = next(pg.load_datasets_graph(["graph5"]))
    for _ in supported_backends():
        test_result = pg.Ordinals(pg.Ordinals(pg.Ordinals(pg.PageRank(normalization='col')))).rank(graph, {"A": 1}) # three ordinal transformations are the same as one
        assert test_result["A"] == 1


def test_sweep():
    _, graph, group = next(pg.load_datasets_one_community(["bigraph"]))
    for _ in supported_backends():
        training, evaluation = pg.split(list(group), training_samples=0.1)
        auc1 = pg.AUC({v: 1 for v in evaluation}, exclude=training).evaluate(pg.Sweep(pg.PageRank()).rank(graph, {v: 1 for v in training}))
        auc2 = pg.AUC({v: 1 for v in evaluation}, exclude=training).evaluate(pg.PageRank().rank(graph, {v: 1 for v in training}))
        assert auc1 > auc2


def test_threshold():
    _, graph, group = next(pg.load_datasets_one_community(["bigraph"]))
    for _ in supported_backends():
        training, evaluation = pg.split(list(group), training_samples=0.5)
        cond1 = pg.Conductance().evaluate(pg.Threshold(pg.Sweep(pg.PageRank())).rank(graph, {v: 1 for v in training}))
        cond2 = pg.Conductance().evaluate(pg.Threshold("gap").transform(pg.PageRank().rank(graph, {v: 1 for v in training}))) # try all api types
        assert cond1 <= cond2
