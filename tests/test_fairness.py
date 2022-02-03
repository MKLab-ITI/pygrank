import pygrank as pg
import pytest


def test_fair_personalizer():
    H = pg.PageRank(assume_immutability=True, normalization="symmetric")
    algorithms = {
        "FairPers": lambda G, p, s: pg.Normalize(pg.FairPersonalizer(H, error_type=pg.Mabs, max_residual=0)).rank(G, p, sensitive=s),
        "FairPers-C": lambda G, p, s: pg.Normalize
            (pg.FairPersonalizer(H, .80, pRule_weight=10, error_type=pg.Mabs, max_residual=0)).rank(G, p, sensitive=s),
        "FairPersSkew": lambda G, p, s: pg.Normalize(pg.FairPersonalizer(H, error_skewing=True, max_residual=0)).rank(G, p, sensitive=s),
        "FairPersSkew-C": lambda G, p, s: pg.Normalize(pg.FairPersonalizer(H, .80, error_skewing=True, pRule_weight=10, max_residual=0)).rank
            (G, p, sensitive=s),
    }
    _, graph, groups = next(pg.load_datasets_multiple_communities(["bigraph"]))
    labels = pg.to_signal(graph, groups[0])
    sensitive = pg.to_signal(graph, groups[1])
    for algorithm in algorithms.values():
        ranks = algorithm(graph, labels, sensitive)
        assert pg.pRule(sensitive)(ranks) > 0.79  # allow a leeway for generalization capabilities compared to 80%


def test_fair_personalizer_mistreatment():
    H = pg.PageRank(assume_immutability=True, normalization="symmetric")
    algorithms = {
        "Base": lambda G, p, s: H.rank(G, p),
        "FairPersMistreat": pg.Normalize(pg.FairPersonalizer(H, parity_type="mistreatment", pRule_weight=10)),
        "FairPersTPR": pg.Normalize(pg.FairPersonalizer(H, parity_type="TPR", pRule_weight=10)),
        "FairPersTNR": pg.Normalize(pg.FairPersonalizer(H, parity_type="TNR", pRule_weight=-1))  # TNR optimization increases mistreatment for this example
    }
    mistreatment = lambda known_scores, sensitive_signal, exclude: \
        pg.AM([pg.Disparity([pg.TPR(known_scores, exclude=1 - (1 - exclude.np) * sensitive_signal.np),
                             pg.TPR(known_scores, exclude=1 - (1 - exclude.np) * (1 - sensitive_signal.np))]),
               pg.Disparity([pg.TNR(known_scores, exclude=1 - (1 - exclude.np) * sensitive_signal.np),
                             pg.TNR(known_scores, exclude=1 - (1 - exclude.np) * (1 - sensitive_signal.np))])])
    _, graph, groups = next(pg.load_datasets_multiple_communities(["synthfeats"]))
    labels = pg.to_signal(graph, groups[0])
    sensitive = pg.to_signal(graph, groups[1])
    train, test = pg.split(labels)
    # TODO: maybe try to exceed 0.8 fairness on this dataset (instead of marginal improvement to just over 0.08)
    base_mistreatment = mistreatment(test, sensitive, train)(algorithms["Base"](graph, train, sensitive))
    for algorithm in algorithms.values():
        if algorithm != algorithms["Base"]:
            print(algorithm.cite())
            assert base_mistreatment < mistreatment(test, sensitive, train)(algorithm(graph, train, sensitive))
    #for algorithm in algorithms.values():
        #print(mistreatment(test, sensitive, train)(algorithm(graph, train, sensitive)))


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
        assert pg.pRule(sensitive)(ranks) > 0.6  # TODO: Check why fairwalk fails by that much and increase the limit.
    sensitive = 1 - sensitive.np
    for algorithm in algorithms.values():
        ranks = algorithm(graph, labels, sensitive)
        assert pg.pRule(sensitive)(ranks) > 0.6


def test_invalid_fairness_arguments():
    _, graph, groups = next(pg.load_datasets_multiple_communities(["bigraph"]))
    labels = pg.to_signal(graph, groups[0])
    sensitive = pg.to_signal(graph, groups[1])
    H = pg.PageRank(assume_immutability=True, normalization="symmetric")
    with pytest.raises(Exception):
        # this tests that a deprecated way of applying fairwalk actually raises an exception
        pg.AdHocFairness(H, method="FairWalk").rank(graph, labels, sensitive=sensitive)
    with pytest.raises(Exception):
        pg.FairPersonalizer(H, parity_type="universal").rank(graph, labels, sensitive=sensitive)
    with pytest.raises(Exception):
        pg.FairWalk(None).transform(H.rank(graph, labels), sensitive=sensitive)
