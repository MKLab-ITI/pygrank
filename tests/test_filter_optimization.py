import pygrank as pg
import pytest
from .test_core import supported_backends


def test_venuerank():
    graph = next(pg.load_datasets_graph(["bigraph"]))
    for _ in supported_backends():
        venuerank = pg.PageRank(alpha=0.85, max_iters=10000, converge_to_eigenvectors=True, tol=1.E-12)
        venuerank_result = venuerank.rank(graph)
        small_restart = pg.PageRank(alpha=0.99, max_iters=10000, tol=1.E-12)
        small_restart_result = small_restart.rank(graph)
        #assert venuerank.convergence.iteration < small_restart.convergence.iteration / 2
        corr = pg.SpearmanCorrelation(pg.Ordinals()(venuerank_result))(pg.Ordinals()(small_restart_result))
        assert corr > 0.99


def test_rank_order_convergence():
    graph = next(pg.load_datasets_graph(["graph9"]))
    algorithm1 = pg.Ordinals(pg.PageRank(0.85, tol=1.E-12, max_iters=1000))
    algorithm2 = pg.Ordinals(pg.PageRank(0.85, convergence=pg.RankOrderConvergenceManager(0.85)))
    algorithm3 = pg.Ordinals(pg.PageRank(0.85, convergence=pg.RankOrderConvergenceManager(0.85, 0.99, "fraction_of_walks")))
    for _ in supported_backends():
        ranks1 = algorithm1.rank(graph, {"A": 1})
        ranks2 = algorithm2.rank(graph, {"A": 1})
        ranks3 = algorithm3.rank(graph, {"A": 1})
        assert pg.SpearmanCorrelation(ranks1)(ranks2) > 0.95
        assert pg.SpearmanCorrelation(ranks1)(ranks3) > 0.95
        assert pg.SpearmanCorrelation(ranks3)(ranks2) > 0.95
        assert "17 iterations" in str(algorithm3.ranker.convergence)
        with pytest.raises(Exception):
            algorithm = pg.Ordinals(pg.PageRank(0.85, convergence=pg.RankOrderConvergenceManager(0.85, 0.99, "unknown")))
            algorithm.rank(graph, {"A": 1})


def test_lanczos_speedup():
    graph = next(pg.load_datasets_graph(["graph9"]))
    #  TODO: implement krylov space for tensorflow
    for algorithm in [pg.HeatKernel]:
        result = pg.Normalize(algorithm(normalization='symmetric')).rank(graph)
        result_lanczos = pg.Normalize(algorithm(normalization='symmetric', krylov_dims=5)).rank(graph,
                                                                                                personalization=None)
        assert pg.Mabs(result)(result_lanczos) < 1  #  TODO: fix optimization for stronger approximation


def test_chebyshev():
    _, graph, group = next(pg.load_datasets_one_community(["bigraph"]))
    #  do not test with tensorflow, as it can be too slow
    training, evaluation = pg.split(pg.to_signal(graph, {v: 1 for v in group}))
    tuned_auc= pg.AUC(evaluation, training).evaluate(pg.ParameterTuner().rank(graph, training))
    tuned_chebyshev_auc= pg.AUC(evaluation, training).evaluate(pg.ParameterTuner(coefficient_type="chebyshev").rank(graph, training))
    assert (tuned_auc-tuned_chebyshev_auc) < 0.1
