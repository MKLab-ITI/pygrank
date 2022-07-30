import pygrank as pg
import pytest
from .test_core import supported_backends


def test_split():
    data = {"community1": ["A", "B", "C", "D"], "community2": ["B", "E", "F", "G", "H", "I"]}
    training, test = pg.split(data, 1)
    assert training == test
    training, test = pg.split(data, 0.5)
    assert len(training["community2"]) == 3
    assert len(training["community1"]) == 2
    assert len(test["community2"]) == 3
    assert len(set(training["community1"])-set(test["community1"])) == len(training["community1"])
    assert len(set(training["community2"])-set(test["community2"])) == len(training["community2"])
    training, test = pg.split(data, 2)
    assert len(training["community2"]) == 2
    assert len(test["community1"]) == 2
    training, test = pg.split(data["community1"], 0.75)
    assert len(training) == 3
    assert len(test) == 1
    training, test = pg.split(set(data["community1"]), 0.75)
    assert len(training) == 3
    assert len(test) == 1


def test_auc_ndcg_compliance():
    _, graph, group = next(pg.load_datasets_one_community(["bigraph"]))
    training, test = pg.split(group, 0.5)
    for _ in supported_backends():
        scores1 = pg.PageRank()(graph, training)
        scores2 = pg.HeatKernel()(graph, training)
        AUC1 = pg.AUC(test, exclude=training)(scores1)
        AUC2 = pg.AUC(test, exclude=training)(scores2)
        NDCG1 = float(pg.NDCG(test, exclude=training)(scores1))
        NDCG2 = float(pg.NDCG(test, exclude=training)(scores2))
        assert (AUC1 < AUC2) == (NDCG1 < NDCG2)
        with pytest.raises(Exception):
            pg.AUC(test, exclude=test, k=len(graph)+1)(scores2)
        with pytest.raises(Exception):
            pg.NDCG(test, exclude=training, k=len(graph)+1)(scores2)


def test_edge_cases():
    assert pg.pRule([0])([0]) == 0
    assert pg.Cos([0])([0]) == 0
    with pytest.raises(Exception):
        pg.Measure()([0, 1, 0])
    with pytest.raises(Exception):
        pg.AUC([0, 0, 0])([0, 1, 0])
    with pytest.raises(Exception):
        pg.AUC([1, 1, 1])([0, 1, 0])
    with pytest.raises(Exception):
        pg.KLDivergence([0], exclude={"A": 1})([1])
    with pytest.raises(Exception):
        pg.Conductance(next(pg.load_datasets_graph(["graph5"])), max_rank=0.5)([1, 1, 1, 1, 1])
    import networkx as nx
    for _ in supported_backends():
        assert pg.Conductance(nx.Graph())([]) == float("inf")  # this is indeed correct in python
        assert pg.Density(nx.Graph())([]) == 0
        assert pg.Modularity(nx.Graph())([]) == 0
        assert pg.KLDivergence([0,1,0])([0,1,0]) == 0
        assert pg.MKLDivergence([0,1,0])([0,1,0]) == 0
        assert pg.KLDivergence([0])([-1]) == 0


def test_strange_input_types():
    _, graph, group = next(pg.load_datasets_one_community(["bigraph"]))
    training, test = pg.split(group)
    for _ in supported_backends():
        scores = pg.PageRank()(graph, {v: 1 for v in training})
        ndcg = pg.NDCG(pg.to_signal(scores, {v: 1 for v in test}), k=3)({v: scores[v] for v in scores})
        ndcg_biased = pg.NDCG(pg.to_signal(scores, {v: 1 for v in test}), k=3)({v: scores[v] for v in test})
        assert ndcg < ndcg_biased


def test_correlation_compliance():
    graph = next(pg.load_datasets_graph(["graph5"]))
    # TODO: Make spearman and pearson correlation support tensorflow
    alg1 = pg.PageRank(alpha=0.5)
    alg2 = pg.PageRank(alpha=0.99)
    pearson_ordinals = pg.PearsonCorrelation(pg.Ordinals(alg1)(graph))(pg.Ordinals(alg2)(graph))
    spearman = pg.SpearmanCorrelation(alg1(graph))(alg2(graph))
    assert pearson_ordinals == spearman


def test_best_direction():
    assert pg.Conductance().best_direction() == -1
    assert pg.Density().best_direction() == 1
    assert pg.Modularity().best_direction() == 1
    assert pg.AUC([1, 2, 3]).best_direction() == 1
    assert pg.Cos([1, 2, 3]).best_direction() == 1
    assert pg.Dot([1, 2, 3]).best_direction() == 1
    assert pg.TPR([1, 2, 3]).best_direction() == 1
    assert pg.TNR([1, 2, 3]).best_direction() == 1
    assert pg.Mabs([1, 2, 3]).best_direction() == -1
    assert pg.MSQ([1, 2, 3]).best_direction() == -1
    assert pg.Euclidean([1, 2, 3]).best_direction() == -1
    assert pg.L2([1, 2, 3]).best_direction() == -1


def test_computations():
    for _ in supported_backends():
        assert pg.Accuracy([1, 2, 3])([1, 2, 3]) == 1
        assert pg.Mabs([3, 1, 1])([2, 0, 2]) == 1
        assert pg.CrossEntropy([1, 1, 1])([1, 1, 1]) < 1.E-12
        assert float(pg.Cos([2, 0, 1])([2, 0, 1])) == 1
        assert float(pg.Cos([2, 0, 1])([-2, 0, -1])) == -1
        assert float(pg.Cos([0, 0, 0])([0, 0, 0])) == 0
        assert float(pg.Dot([1, 1, 1])([1, 1, 1])) == 3
        assert float(pg.TPR([1, 0, 0, 0])([1, 1, 0, 0])) == 1
        assert float(pg.PPV([1, 0, 0, 0])([1, 1, 0, 0])) == 0.5
        assert float(pg.TNR([1, 0, 0, 1])([1, 1, 0, 0])) == 0.5
        assert float(pg.Euclidean([0, 0, 0, 1])([1, 1, 0, 0])) < float(pg.Euclidean([0, 0, 0, 1])([1, 1, 1, 0]))


def test_aggregated():
    y1 = [1, 1, 0]
    y2 = [1, 0, 0]
    y3 = [1, 1, 0]
    for _ in supported_backends():
        # TODO: investigate why not exactly the same always (numerical precision should be lower for numpy)
        epsilon = 1.E-6
        assert abs(float(pg.GM().add(pg.AUC(y1), max_val=0.5).add(pg.AUC(y2), min_val=0.9).evaluate(y3)) - 0.45**0.5) < epsilon
        assert abs(float(pg.AM().add(pg.AUC(y1), max_val=0.5).add(pg.AUC(y2), min_val=0.9).evaluate(y3)) - 0.7) < epsilon
        assert abs(float(pg.Disparity().add(pg.AUC(y1), max_val=0.5).add(pg.AUC(y2), min_val=0.9).evaluate(y3))-0.4) < epsilon
        assert abs(float(pg.Disparity().add(pg.AUC(y1), max_val=0.5).add(pg.AUC(y2), min_val=0.9).evaluate(y3))
                   + float(pg.Parity().add(pg.AUC(y1), max_val=0.5).add(pg.AUC(y2), min_val=0.9).evaluate(y3)-1)) < epsilon


def test_remove_edges():
    import networkx as nx
    graph = next(pg.load_datasets_graph(["graph5"], graph_api=nx))
    # TODO: make removing edges possible for fastgraph
    assert graph.has_edge("A", "B")
    assert graph.has_edge("C", "D")
    pg.remove_intra_edges(graph, {"community1": ["A", "B"], "community2": ["D", "C"]})
    assert graph.has_edge("B", "C")
    assert not graph.has_edge("A", "B")
    assert not graph.has_edge("C", "D")
