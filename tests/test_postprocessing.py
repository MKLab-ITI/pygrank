import unittest
import pygrank as pg
from experiments.importer import fairness_dataset
from tests.example_graph import test_graph, test_block_model_graph


class Test(unittest.TestCase):

    def test_autotune(self):
        G, groups = test_block_model_graph()
        group = groups[0]
        training, evaluation = pg.split(pg.to_signal(G, {v: 1 for v in group}), training_samples=0.5)
        auc1 = pg.AUC(evaluation, exclude=training)(pg.PageRank().rank(training))
        auc2 = pg.AUC(evaluation, exclude=training)(pg.ParameterTuner(optimization_dict=dict()).rank(training))
        self.assertLessEqual(auc1, auc2, "Autotune should find good parameters")

    def test_autotune_manual(self):
        G, groups = test_block_model_graph()
        group = groups[0]
        training, evaluation = pg.split(pg.to_signal(G, {v: 1 for v in group}), training_samples=0.5)
        auc1 = pg.AUC(evaluation, exclude=training)(pg.PageRank().rank(training))
        alg2 = pg.ParameterTuner(lambda params: pg.PageRank(params[0]), max_vals=[0.99], min_vals=[0.5]).tune(training)
        auc2 = pg.AUC(evaluation, exclude=training)(alg2.rank(training))
        self.assertLessEqual(auc1, auc2, "Autotune should find good parameters")

    def test_autotune_methods(self):
        G, groups = test_block_model_graph()
        group = groups[0]
        training, evaluation = pg.split(pg.to_signal(G, {v: 1 for v in group}), training_samples=0.5)

        auc1 = max(pg.AUC(evaluation, exclude=training)(ranker.rank(training)) for ranker in pg.create_demo_filters().values())
        auc2 = pg.AUC(evaluation, exclude=training)(pg.AlgorithmSelection().rank(training))
        self.assertLessEqual(auc1, auc2, "Autotune should find best method")

    def test_tautology(self):
        G = test_graph()
        r = pg.PageRank().rank(G)
        tr = pg.Tautology(pg.PageRank()).rank(G)
        rt = pg.Tautology().transform(r)
        for u in G:
            self.assertEqual(r[u], rt[u])
            self.assertEqual(r[u], tr[u])

        u = pg.Tautology().rank(G)
        self.assertEqual(float(sum(u.np)), len(G))

    def test_normalize_range(self):
        G = test_graph()
        r = pg.Normalize(pg.PageRank(), "range").rank(G)
        self.assertAlmostEqual(min(r[v] for v in G), 0, places=15)
        self.assertAlmostEqual(max(r[v] for v in G), 1, places=15)

    def test_normalize_sum(self):
        G = test_graph()
        r = pg.Normalize(pg.PageRank(), "sum").rank(G)
        self.assertAlmostEqual(sum(r[v] for v in G), 1, places=15)

    def test_normalize_invalid(self):
        G = test_graph()
        with self.assertRaises(Exception):
            pg.Normalize(pg.PageRank(), "unknown").rank(G)

    def test_transform_primitives(self):
        from pygrank.core.backend import sum
        G = test_graph()
        r1 = pg.Normalize(pg.PageRank(), "sum").rank(G)
        r2 = pg.Transformer(pg.PageRank(), lambda x: x/sum(x)).rank(G)
        for v in G:
            self.assertAlmostEqual(r1[v], r2[v], places=15)

    def test_transform_individuals(self):
        import math
        G = test_graph()
        r1 = pg.Transformer(math.exp).transform(pg.PageRank()(G))
        r2 = pg.Transformer(pg.PageRank(), pg.exp).rank(G)
        for v in G:
            self.assertAlmostEqual(r1[v], r2[v], places=15)

    def test_ordinals(self):
        G = test_graph()
        test_result = pg.Ordinals(pg.Ordinals(pg.Ordinals(pg.PageRank(normalization='col')))).rank(G, {"A": 1}) # three ordinal transformations are the same as one
        self.assertAlmostEqual(test_result["A"], 1, places=15, msg="Ordinals should compute without errors (seed node is highest rank in small graphs)")

    def test_normalization(self):
        G = test_graph()
        test_result = pg.Normalize("sum", pg.PageRank(normalization='col')).rank(G)
        self.assertAlmostEqual(sum(test_result.values()), 1, places=15, msg="Sum normalization should sum to 1")
        test_result = pg.Normalize("max", pg.PageRank(normalization='col')).rank(G)
        self.assertAlmostEqual(max(test_result.values()), 1, places=15, msg="Max should have max 1")
        test_result = pg.Normalize("sum").transform(pg.PageRank(normalization='col').rank(G))
        self.assertAlmostEqual(sum(test_result.values()), 1, places=15, msg="Normalization should be able to use transformations")

    def test_oversampling(self):
        graph = test_graph()
        test_result = pg.SeedOversampling(pg.PageRank(normalization='col')).rank(graph, {"A": 1})
        test_result = pg.SeedOversampling(pg.PageRank(normalization='col'), 'top').rank(graph, {"A": 1})
        test_result = pg.SeedOversampling(pg.PageRank(normalization='col'), 'neighbors').rank(graph, {"A": 1})
        with self.assertRaises(Exception):
            test_result = pg.SeedOversampling(pg.PageRank(normalization='col'), 'unknown').rank(graph, {"A": 1})
        with self.assertRaises(Exception):
            test_result = pg.SeedOversampling(pg.PageRank(normalization='col')).rank(graph, {"A": 0.5, "B": 1})

    def test_boosted_oversampling(self):
        graph = test_graph()
        test_result = pg.BoostedSeedOversampling(pg.PageRank(normalization='col')).rank(graph, {"A": 1})
        test_result = pg.BoostedSeedOversampling(pg.PageRank(normalization='col'), 'naive').rank(graph, {"A": 1})
        test_result = pg.BoostedSeedOversampling(pg.PageRank(normalization='col'), oversample_from_iteration='original').rank(graph, {"A": 1})
        with self.assertRaises(Exception):
            test_result = pg.BoostedSeedOversampling(pg.PageRank(normalization='col'), 'unknown').rank(graph, {"A": 1})
        with self.assertRaises(Exception):
            test_result = pg.BoostedSeedOversampling(pg.PageRank(normalization='col'), oversample_from_iteration='unknown').rank(graph, {"A": 1})

    def test_abstract_postprocessor(self):
        with self.assertRaises(Exception):
            p = pg.Postprocessor(pg.PageRank())
            graph = test_graph()
            p.rank(graph)

    def test_optimizer_errors(self):
        from pygrank.algorithms import optimize
        with self.assertRaises(Exception):
            optimize(loss=lambda p: (p[0]-2)**2+(p[1]-1)**4, max_vals=[5, 5], parameter_tol=1.E-8, divide_range=1)
        with self.assertRaises(Exception):
            optimize(loss=lambda p: (p[0]-2)**2+(p[1]-1)**4, max_vals=[5, 5], min_vals=[5, 6])

    def test_sweep(self):
        import random
        graph, groups = test_block_model_graph(nodes=600, seed=1)
        group = groups[0]
        random.seed(1)
        training, evaluation = pg.split(list(group), training_samples=0.5)
        auc1 = pg.AUC({v: 1 for v in evaluation}, exclude=training).evaluate(pg.Sweep(pg.PageRank()).rank(graph, {v: 1 for v in training}))
        auc2 = pg.AUC({v: 1 for v in evaluation}, exclude=training).evaluate(pg.PageRank().rank(graph, {v: 1 for v in training}))
        self.assertLess(auc2+0.2, auc1, "The Sweep procedure should significantly improve AUC")

    def test_threshold(self):
        graph, groups = test_block_model_graph(nodes=600, seed=2)
        group = groups[0]
        training, evaluation = pg.split(list(group), training_samples=0.5)
        cond1 = pg.Conductance().evaluate(pg.Threshold(pg.Sweep(pg.PageRank())).rank(graph, {v: 1 for v in training}))
        cond2 = pg.Conductance().evaluate(pg.Threshold("gap").transform(pg.PageRank().rank(graph, {v: 1 for v in training}))) # try both versions
        self.assertLess(cond2*4.5, cond1, "The Sweep procedure should significantly reduce conductance after gap thresholding")

    def test_optimizer(self):
        # a simple function
        p = pg.optimize(loss=lambda p: (p[0]-2)**2+(p[1]-1)**4, max_vals=[5, 5], parameter_tol=1.E-8)
        self.assertAlmostEqual(p[0], 2, places=6, msg="Optimizer should easily optimize a convex function", verbose=True)
        self.assertAlmostEqual(p[1], 1, places=6, msg="Optimizer should easily optimize a convex function")

        # a simple function with redundant inputs and tol instead of parameter tolerance
        p = pg.optimize(loss=lambda p: (p[0]-2)**2+(p[1]-1)**4, max_vals=[5, 5, 5], min_vals=[0, 0, 5], deviation_tol=1.E-6, divide_range="shrinking")
        self.assertAlmostEqual(p[0], 2, places=3, msg="Optimizer should easily optimize a convex function")
        self.assertAlmostEqual(p[1], 1, places=3, msg="Optimizer should easily optimize a convex function")

        # https://en.wikipedia.org/wiki/Test_functions_for_optimization

        # Beale function
        p = pg.optimize(loss=lambda p: (1.5-p[0]+p[0]*p[1])**2+(2.25-p[0]+p[0]*p[1]**2)**2+(2.625-p[0]+p[0]*p[1]**3)**2, max_vals=[4.5, 4.5], min_vals=[-4.5, -4.5], parameter_tol=1.E-8)
        self.assertAlmostEqual(p[0], 3, places=6, msg="Optimizer should optimize the Beale function")
        self.assertAlmostEqual(p[1], 0.5, places=6, msg="Optimizer should optimize the Beale function")

        # Booth function
        p = pg.optimize(loss=lambda p: (p[0]+2*p[1]-7)**2+(2*p[0]+p[1]-5)**2, max_vals=[10, 10], min_vals=[-10, -10], parameter_tol=1.E-8)
        self.assertAlmostEqual(p[0], 1, places=6, msg="Optimizer should optimize the Booth function")
        self.assertAlmostEqual(p[1], 3, places=6, msg="Optimizer should optimize the Booth function")

        # Beale function with depth instead of small divide range
        p = pg.optimize(loss=lambda p: (1.5-p[0]+p[0]*p[1])**2+(2.25-p[0]+p[0]*p[1]**2)**2+(2.625-p[0]+p[0]*p[1]**3)**2,
                     max_vals=[4.5, 4.5], min_vals=[-4.5, -4.5], parameter_tol=1.E-8, divide_range=2, depth=100)
        self.assertAlmostEqual(p[0], 3, places=6, msg="Optimizer should optimize the Beale function")
        self.assertAlmostEqual(p[1], 0.5, places=6, msg="Optimizer should optimize the Beale function")

    def test_fair_personalizer(self):
        H = pg.PageRank(assume_immutability=True, normalization="symmetric")
        algorithms = {
            "FairPers": lambda G, p, s: pg.Normalize(pg.FairPersonalizer(H, error_type=pg.Mabs, max_residual=0)).rank(G, p, sensitive=s),
            "FairPers-C": lambda G, p, s: pg.Normalize(pg.FairPersonalizer(H, .80, pRule_weight=10, error_type=pg.Mabs, max_residual=0)).rank(G, p, sensitive=s),
            "FairPersKL": lambda G, p, s: pg.Normalize(pg.FairPersonalizer(H, max_residual=0)).rank(G, p, sensitive=s),
            "FairPersKL-C": lambda G, p, s: pg.Normalize(pg.FairPersonalizer(H, .80, pRule_weight=10, max_residual=0)).rank(G, p, sensitive=s),
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
        sensitive.np = 1-sensitive.np
        for algorithm in algorithms.values():
            ranks = algorithm(graph, labels, sensitive)
            self.assertGreater(pg.pRule(sensitive.np)(ranks), 0.9, "should satisfy fairness requirements")

    def test_fairwalk_invalid(self):
        graph, sensitive, labels = fairness_dataset("facebook0", 0, sensitive_group=1, path="data/")
        H = pg.PageRank(assume_immutability=True, normalization="symmetric")
        with self.assertRaises(Exception):
            pg.AdHocFairness(H, method="FairWalk").rank(graph, labels, sensitive=sensitive)
        with self.assertRaises(Exception):
            pg.FairWalk(None).transform(H.rank(graph, labels), sensitive=sensitive)
