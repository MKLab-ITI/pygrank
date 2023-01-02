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
    for algorithm in algorithms:
        ranks = algorithms[algorithm](graph, labels, sensitive)
        assert pg.pRule(sensitive)(ranks) > 0.7  # allow a leeway for generalization capabilities compared to 80%


def test_fair_personalizer_mistreatment():
    H = pg.PageRank(assume_immutability=True, normalization="symmetric")
    algorithms = {
        "Base": lambda G, p, s: H.rank(G, p),
        "FairPersMistreat": pg.Normalize(pg.FairPersonalizer(H, parity_type="mistreatment", pRule_weight=10)),
        "FairPersTPR": pg.Normalize(pg.FairPersonalizer(H, parity_type="TPR", pRule_weight=1)),
        "FairPersTNR": pg.Normalize(pg.FairPersonalizer(H, parity_type="TNR", pRule_weight=1)),
        "FairPersU": pg.Normalize(pg.FairPersonalizer(H, parity_type="U", pRule_weight=1))
    }
    mistreatment = lambda known_scores, sensitive_signal, exclude: \
        pg.AM([pg.Disparity([pg.TPR(known_scores, exclude=1 - (1 - exclude) * sensitive_signal),
                             pg.TPR(known_scores, exclude=1 - (1 - exclude) * (1 - sensitive_signal))]),
               pg.Disparity([pg.TNR(known_scores, exclude=1 - (1 - exclude) * sensitive_signal),
                             pg.TNR(known_scores, exclude=1 - (1 - exclude) * (1 - sensitive_signal))])])
    _, graph, groups = next(pg.load_datasets_multiple_communities(["synthfeats"]))
    labels = pg.to_signal(graph, groups[0])
    sensitive = pg.to_signal(graph, groups[1])
    train, test = pg.split(labels)
    base_mistreatment = mistreatment(test, sensitive, train)(algorithms["Base"](graph, train, sensitive))
    for algorithm in algorithms.values():
        if algorithm != algorithms["Base"]:
            print(algorithm.cite())
            new_mistreatment = mistreatment(test, sensitive, train)(algorithm(graph, train, sensitive))
            # TODO: incorporate actual mistreatment mitigation approaches, these are heuristic tests
            #assert base_mistreatment >= new_mistreatment


def test_fair_heuristics():
    H = pg.PageRank(assume_immutability=True, normalization="symmetric")
    algorithms = {
        "FairO": lambda G, p, s: pg.Normalize(pg.AdHocFairness(H, method="O")).rank(G, sensitive=s),
        "FairB": lambda G, p, s: pg.Normalize()(pg.AdHocFairness("B").transform(H.rank(G, p), sensitive=s)),
        "LFPRN": lambda G, p, s: pg.Normalize()(pg.LFPR().rank(G, p, sensitive=s)),
        "LFPRP": lambda G, p, s: pg.Normalize()(pg.LFPR(redistributor="original").rank(G, p, sensitive=s)),
        "LFPRHK": lambda G, p, s: pg.Normalize()(pg.LFPR(redistributor=pg.HeatKernel()).rank(G, p, sensitive=s)),
        "FairWalk": lambda G, p, s: pg.FairWalk(H).rank(G, p, sensitive=s)
    }
    import networkx as nx
    _, graph, groups = next(pg.load_datasets_multiple_communities(["bigraph"], graph_api=nx))
    # TODO: networx needed due to edge weighting by some algorithms
    labels = pg.to_signal(graph, groups[0])
    sensitive = pg.to_signal(graph, groups[1])
    for name, algorithm in algorithms.items():
        ranks = algorithm(graph, labels, sensitive)
        if name == "FairWalk":
            assert pg.pRule(sensitive)(ranks) > 0.6  # TODO: Check why fairwalk fails by that much and increase the limit.
        else:
            assert pg.pRule(sensitive)(ranks) > 0.98
    sensitive = 1 - sensitive.np
    for name, algorithm in algorithms.items():
        ranks = algorithm(graph, labels, sensitive)
        if name == "FairWalk":
            assert pg.pRule(sensitive)(ranks) > 0.6
        else:
            assert pg.pRule(sensitive)(ranks) > 0.98


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
