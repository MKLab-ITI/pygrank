import pygrank as pg
import pytest


def test_fair_personalizer():
    H = pg.PageRank(assume_immutability=True, normalization="symmetric")
    algorithms = {
        "FairPers": lambda G, p, s: pg.Normalize(pg.FairPersonalizer(H, error_type=pg.Mabs, max_residual=0)).rank(G, p, sensitive=s),
        "FairPers-C": lambda G, p, s: pg.Normalize
            (pg.FairPersonalizer(H, .80, pRule_weight=10, error_type=pg.Mabs, max_residual=0)).rank(G, p, sensitive=s),
        "FairPersKL": lambda G, p, s: pg.Normalize(pg.FairPersonalizer(H, max_residual=0)).rank(G, p, sensitive=s),
        "FairPersKL-C": lambda G, p, s: pg.Normalize(pg.FairPersonalizer(H, .80, pRule_weight=10, max_residual=0)).rank
            (G, p, sensitive=s),
    }
    _, graph, groups = next(pg.load_datasets_multiple_communities(["bigraph"]))
    labels = pg.to_signal(graph, groups[0])
    sensitive = pg.to_signal(graph, groups[1])
    for algorithm in algorithms.values():
        ranks = algorithm(graph, labels, sensitive)
        assert pg.pRule(sensitive)(ranks) > 0.8


def test_fair_heuristics():
    H = pg.PageRank(assume_immutability=True, normalization="symmetric")
    algorithms = {
        "FairO": lambda G, p, s: pg.Normalize(pg.AdHocFairness(H, method="O")).rank(G, sensitive=s),
        "FairB": lambda G, p, s: pg.Normalize()(pg.AdHocFairness("B").transform(H.rank(G, p), sensitive=s)),
        "FairWalk": lambda G, p, s: pg.FairWalk(H).rank(G, p, sensitive=s)
    }

    _, graph, groups = next(pg.load_datasets_multiple_communities(["bigraph"]))
    labels = pg.to_signal(graph, groups[0])
    sensitive = pg.to_signal(graph, groups[1])
    for algorithm in algorithms.values():
        ranks = algorithm(graph, labels, sensitive)
        assert pg.pRule(sensitive)(ranks) > 0.6  #  TODO: Check why this fairwalk fails that much and increase the limit.
    sensitive = 1- sensitive.np
    for algorithm in algorithms.values():
        ranks = algorithm(graph, labels, sensitive)
        pg.pRule(sensitive.np)(ranks) > 0.6


def test_fairwalk_invalid():
    _, graph, groups = next(pg.load_datasets_multiple_communities(["bigraph"]))
    labels = pg.to_signal(graph, groups[0])
    sensitive = pg.to_signal(graph, groups[1])
    H = pg.PageRank(assume_immutability=True, normalization="symmetric")
    with pytest.raises(Exception):
        pg.AdHocFairness(H, method="FairWalk").rank(graph, labels, sensitive=sensitive)
    with pytest.raises(Exception):
        pg.FairWalk(None).transform(H.rank(graph, labels), sensitive=sensitive)