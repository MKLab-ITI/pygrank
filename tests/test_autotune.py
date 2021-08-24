import pygrank as pg
import pytest


def test_optimizer_errors():
    from pygrank.algorithms import optimize
    with pytest.raises(Exception):
        optimize(loss=lambda p: (p[0] - 2) ** 2 + (p[1] - 1) ** 4, max_vals=[5, 5], parameter_tol=1.E-8, divide_range=1)
    with pytest.raises(Exception):
        optimize(loss=lambda p: (p[0] - 2) ** 2 + (p[1] - 1) ** 4, max_vals=[5, 5], min_vals=[5, 6])


def test_optimizer():
    # https://en.wikipedia.org/wiki/Test_functions_for_optimization

    # a simple function
    p = pg.optimize(loss=lambda p: (p[0]-2)**2+(p[1]-1)**4, max_vals=[5, 5], parameter_tol=1.E-8)
    assert abs(p[0]-2) < 1.E-6
    assert abs(p[1]-1) < 1.E-6

    # a simple function with redundant inputs and tol instead of parameter tolerance
    p = pg.optimize(loss=lambda p: (p[0]-2)**2+(p[1]-1)**4,
                    max_vals=[5, 5, 5], min_vals=[0, 0, 5], deviation_tol=1.E-6, divide_range="shrinking")
    assert abs(p[0]-2) < 1.E-3
    assert abs(p[1]-1) < 1.E-3

    # Beale function
    p = pg.optimize(loss=lambda p: (1.5-p[0]+p[0]*p[1])**2+(2.25-p[0]+p[0]*p[1]**2)**2+(2.625-p[0]+p[0]*p[1]**3)**2,
                    max_vals=[4.5, 4.5], min_vals=[-4.5, -4.5], parameter_tol=1.E-8)
    assert abs(p[0]-3) < 1.E-6
    assert abs(p[1]-0.5) < 1.E-6

    # Booth function
    p = pg.optimize(loss=lambda p: (p[0]+2*p[1]-7)**2+(2*p[0]+p[1]-5)**2,
                    max_vals=[10, 10], min_vals=[-10, -10], parameter_tol=1.E-8)
    assert abs(p[0] - 1) < 1.E-6
    assert abs(p[1] - 3) < 1.E-6

    # Beale function with depth instead of small divide range
    p = pg.optimize(loss=lambda p: (1.5-p[0]+p[0]*p[1])**2+(2.25-p[0]+p[0]*p[1]**2)**2+(2.625-p[0]+p[0]*p[1]**3)**2,
                    max_vals=[4.5, 4.5], min_vals=[-4.5, -4.5], parameter_tol=1.E-8, divide_range=2, depth=100)
    assert abs(p[0] - 3) < 1.E-6
    assert abs(p[1] - 0.5) < 1.E-6


def test_autotune():
    _, G, groups = next(pg.load_datasets_multiple_communities(["bigraph"]))
    group = groups[0]
    training, evaluation = pg.split(pg.to_signal(G, {v: 1 for v in group}), training_samples=0.5)
    auc1 = pg.AUC(evaluation, exclude=training)(pg.PageRank().rank(training))
    auc2 = pg.AUC(evaluation, exclude=training)(pg.ParameterTuner(optimization_dict=dict()).rank(training))
    assert auc1 <= auc2


def test_autotune_manual():
    _, G, groups = next(pg.load_datasets_multiple_communities(["bigraph"]))
    group = groups[0]
    training, evaluation = pg.split(pg.to_signal(G, {v: 1 for v in group}), training_samples=0.5)
    auc1 = pg.AUC(evaluation, exclude=training)(pg.PageRank().rank(training))
    alg2 = pg.ParameterTuner(lambda params: pg.PageRank(params[0]), max_vals=[0.99], min_vals=[0.5]).tune(training)
    auc2 = pg.AUC(evaluation, exclude=training)(alg2.rank(training))
    assert auc1 <= auc2


def test_autotune_methods():
    import numpy as np
    _, G, groups = next(pg.load_datasets_multiple_communities(["bigraph"]))
    group = groups[0]
    training, evaluation = pg.split(pg.to_signal(G, {v: 1 for v in group}))
    aucs = [pg.AUC(evaluation, exclude=training)(ranker.rank(training)) for ranker in pg.create_demo_filters().values()]
    auc2 = pg.AUC(evaluation, exclude=training)(pg.AlgorithmSelection().rank(training))
    assert max(aucs)-np.std(aucs) <= auc2


def test_autotune_backends():
    _, G, groups = next(pg.load_datasets_multiple_communities(["bigraph"]))
    group = groups[0]
    training, evaluation = pg.split(pg.to_signal(G, {v: 1 for v in group}), training_samples=0.5)
    auc1 = pg.AUC(evaluation, exclude=training)(pg.AlgorithmSelection(measure=pg.KLDivergence).rank(training))
    auc2 = pg.AUC(evaluation, exclude=training)(pg.AlgorithmSelection(measure=pg.KLDivergence, tuning_backend="tensorflow").rank(training))
    assert auc1 == auc2