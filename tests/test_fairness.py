
def test_fair_personalizer(self):
    H = pg.PageRank(assume_immutability=True, normalization="symmetric")
    algorithms = {
        "FairPers": lambda G, p, s: pg.Normalize(pg.FairPersonalizer(H, error_type=pg.Mabs, max_residual=0)).rank(G, p, sensitive=s),
        "FairPers-C": lambda G, p, s: pg.Normalize
            (pg.FairPersonalizer(H, .80, pRule_weight=10, error_type=pg.Mabs, max_residual=0)).rank(G, p, sensitive=s),
        "FairPersKL": lambda G, p, s: pg.Normalize(pg.FairPersonalizer(H, max_residual=0)).rank(G, p, sensitive=s),
        "FairPersKL-C": lambda G, p, s: pg.Normalize(pg.FairPersonalizer(H, .80, pRule_weight=10, max_residual=0)).rank
            (G, p, sensitive=s),
    }
    graph, sensitive, labels = fairness_dataset("facebook0", 0, sensitive_group=1, path="data/")
    for algorithm in algorithms.values():
        ranks = algorithm(graph, labels, sensitive)
        self.assertGreater(pg.pRule(sensitive)(ranks), 0.8, "should satisfy fairness requirements")

def test_fair_heuristics(self):
    H = pg.PageRank(assume_immutability=True, normalization="symmetric")
    algorithms = {
        "FairO": lambda G, p, s: pg.Normalize(pg.AdHocFairness(H, method="O")).rank(G, sensitive=s),
        "FairB": lambda G, p, s: pg.Normalize()(pg.AdHocFairness("B").transform(H.rank(G, p), sensitive=s)),
        "FairWalk": lambda G, p, s: pg.FairWalk(H).rank(G, p, sensitive=s)
    }
    graph, sensitive, labels = fairness_dataset("facebook0", 0, sensitive_group=1, path="data/")
    for algorithm in algorithms.values():
        ranks = algorithm(graph, labels, sensitive)
        self.assertGreater(pg.pRule(sensitive)(ranks), 0.9, "should satisfy fairness requirements")
    sensitive = pg.to_signal(graph, sensitive)
    sensitive.np = 1- sensitive.np
    for algorithm in algorithms.values():
        ranks = algorithm(graph, labels, sensitive)
        self.assertGreater(pg.pRule(sensitive.np)(ranks), 0.9, "should satisfy fairness requirements")


def test_fairwalk_invalid(self):
    graph, sensitive, labels = fairness_dataset("facebook0", 0, sensitive_group=1, path="data/")
    H = pg.PageRank(assume_immutability=True, normalization="symmetric")
    with pytest.raises(Exception):
        pg.AdHocFairness(H, method="FairWalk").rank(graph, labels, sensitive=sensitive)
    with pytest.raises(Exception):
        pg.FairWalk(None).transform(H.rank(graph, labels), sensitive=sensitive)