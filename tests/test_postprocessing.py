import unittest

from experiments.importer import fairness_dataset
from tests.example_graph import test_graph, test_block_model_graph


class Test(unittest.TestCase):
    def test_autotune(self):
        from pygrank import PageRank, ParameterTuner, AUC, split, to_signal
        G, groups = test_block_model_graph()
        group = groups[0]
        training, evaluation = split(to_signal(G, {v: 1 for v in group}), training_samples=0.5)
        auc1 = AUC(evaluation, exclude=training)(PageRank().rank(training))
        auc2 = AUC(evaluation, exclude=training)(ParameterTuner(optimization_dict=dict()).rank(training))
        self.assertLessEqual(auc1, auc2, "Autotune should find good parameters")

    def test_autotune_manual(self):
        from pygrank import PageRank, ParameterTuner, AUC, split, to_signal
        G, groups = test_block_model_graph()
        group = groups[0]
        training, evaluation = split(to_signal(G, {v: 1 for v in group}), training_samples=0.5)
        auc1 = AUC(evaluation, exclude=training)(PageRank().rank(training))
        alg2 = ParameterTuner(lambda params: PageRank(params[0]), max_vals=[0.99], min_vals=[0.5]).tune(training)
        auc2 = AUC(evaluation, exclude=training)(alg2.rank(training))
        self.assertLessEqual(auc1, auc2, "Autotune should find good parameters")

    def test_tautology(self):
        from pygrank import PageRank, Tautology, sum
        G = test_graph()
        r = PageRank().rank(G)
        tr = Tautology(PageRank()).rank(G)
        rt = Tautology().transform(r)
        for u in G:
            self.assertEqual(r[u], rt[u])
            self.assertEqual(r[u], tr[u])

        u = Tautology().rank(G)
        self.assertEqual(float(sum(u.np)), len(G))

    def test_normalize_range(self):
        from pygrank import PageRank, Normalize
        G = test_graph()
        r = Normalize(PageRank(), "range").rank(G)
        self.assertAlmostEqual(min(r[v] for v in G), 0, places=16)
        self.assertAlmostEqual(max(r[v] for v in G), 1, places=16)

    def test_normalize_sum(self):
        from pygrank import PageRank, Normalize
        G = test_graph()
        r = Normalize(PageRank(), "sum").rank(G)
        self.assertAlmostEqual(sum(r[v] for v in G), 1, places=16)

    def test_normalize_invalid(self):
        from pygrank import PageRank, Normalize
        G = test_graph()
        with self.assertRaises(Exception):
            Normalize(PageRank(), "unknown").rank(G)

    def test_transform_primitives(self):
        from pygrank import PageRank, Normalize, Transformer
        from pygrank.core.backend import sum
        G = test_graph()
        r1 = Normalize(PageRank(), "sum").rank(G)
        r2 = Transformer(PageRank(), lambda x: x/sum(x)).rank(G)
        for v in G:
            self.assertAlmostEqual(r1[v], r2[v], places=16)

    def test_transform_individuals(self):
        from pygrank import PageRank, Transformer
        import math
        from pygrank.core import backend
        G = test_graph()
        r1 = Transformer(math.exp).transform(PageRank()(G))
        r2 = Transformer(PageRank(), backend.exp).rank(G)
        for v in G:
            self.assertAlmostEqual(r1[v], r2[v], places=16)

    def test_ordinals(self):
        from pygrank import PageRank, Ordinals
        G = test_graph()
        test_result = Ordinals(Ordinals(Ordinals(PageRank(normalization='col')))).rank(G, {"A": 1}) # three ordinal transformations are the same as one
        self.assertAlmostEqual(test_result["A"], 1, places=16, msg="Ordinals should compute without errors (seed node is highest rank in small graphs)")

    def test_normalization(self):
        from pygrank import PageRank, Normalize
        G = test_graph()
        test_result = Normalize("sum", PageRank(normalization='col')).rank(G)
        self.assertAlmostEqual(sum(test_result.values()), 1, places=16, msg="Sum normalization should sum to 1")
        test_result = Normalize("max", PageRank(normalization='col')).rank(G)
        self.assertAlmostEqual(max(test_result.values()), 1, places=16, msg="Max should have max 1")
        test_result = Normalize("sum").transform(PageRank(normalization='col').rank(G))
        self.assertAlmostEqual(sum(test_result.values()), 1, places=16, msg="Normalization should be able to use transformations")

    def test_oversampling(self):
        from pygrank import PageRank, SeedOversampling
        G = test_graph()
        test_result = SeedOversampling(PageRank(normalization='col')).rank(G, {"A": 1})
        test_result = SeedOversampling(PageRank(normalization='col'), 'top').rank(G, {"A": 1})
        test_result = SeedOversampling(PageRank(normalization='col'), 'neighbors').rank(G, {"A": 1})
        with self.assertRaises(Exception):
            test_result = SeedOversampling(PageRank(normalization='col'), 'unknown').rank(G, {"A": 1})
        with self.assertRaises(Exception):
            test_result = SeedOversampling(PageRank(normalization='col')).rank(G, {"A": 0.5, "B": 1})

    def test_boosted_oversampling(self):
        from pygrank import PageRank, BoostedSeedOversampling
        G = test_graph()
        test_result = BoostedSeedOversampling(PageRank(normalization='col')).rank(G, {"A": 1})
        test_result = BoostedSeedOversampling(PageRank(normalization='col'), 'naive').rank(G, {"A": 1})
        test_result = BoostedSeedOversampling(PageRank(normalization='col'), oversample_from_iteration='original').rank(G, {"A": 1})
        with self.assertRaises(Exception):
            test_result = BoostedSeedOversampling(PageRank(normalization='col'), 'unknown').rank(G, {"A": 1})
        with self.assertRaises(Exception):
            test_result = BoostedSeedOversampling(PageRank(normalization='col'), oversample_from_iteration='unknown').rank(G, {"A": 1})

    def test_abstract_postprocessor(self):
        from pygrank import PageRank, Postprocessor
        with self.assertRaises(Exception):
            p = Postprocessor(PageRank())
            G = test_graph()
            p.rank(G)

    def test_optimizer_errors(self):
        from pygrank.algorithms import optimize
        with self.assertRaises(Exception):
            optimize(loss=lambda p: (p[0]-2)**2+(p[1]-1)**4, max_vals=[5, 5], parameter_tol=1.E-8, divide_range=1)
        with self.assertRaises(Exception):
            optimize(loss=lambda p: (p[0]-2)**2+(p[1]-1)**4, max_vals=[5, 5], min_vals=[5, 6])

    def test_sweep(self):
        from pygrank import PageRank, Sweep, AUC, split
        import random
        G, groups = test_block_model_graph(nodes=600, seed=1)
        group = groups[0]
        random.seed(1)
        training, evaluation = split(list(group), training_samples=0.5)
        auc1 = AUC({v: 1 for v in evaluation}, exclude=training).evaluate(Sweep(PageRank()).rank(G, {v: 1 for v in training}))
        auc2 = AUC({v: 1 for v in evaluation}, exclude=training).evaluate(PageRank().rank(G, {v: 1 for v in training}))
        self.assertLess(auc2+0.22, auc1, "The Sweep procedure should significantly improve AUC")

    def test_threshold(self):
        from pygrank import PageRank, Threshold, Sweep, Conductance, split
        import random
        G, groups = test_block_model_graph(nodes=600, seed=1)
        group = groups[0]
        random.seed(1)
        training, evaluation = split(list(group), training_samples=0.5)
        cond1 = Conductance().evaluate(Threshold(Sweep(PageRank())).rank(G, {v: 1 for v in training}))
        cond2 = Conductance().evaluate(Threshold("gap").transform(PageRank().rank(G, {v: 1 for v in training}))) # try both versions
        self.assertLess(cond2*4.5, cond1, "The Sweep procedure should significantly reduce conductance after gap thresholding")

    def test_optimizer(self):
        from pygrank.algorithms import optimize

        # a simple function
        p = optimize(loss=lambda p: (p[0]-2)**2+(p[1]-1)**4, max_vals=[5, 5], parameter_tol=1.E-8)
        self.assertAlmostEqual(p[0], 2, places=6, msg="Optimizer should easily optimize a convex function")
        self.assertAlmostEqual(p[1], 1, places=6, msg="Optimizer should easily optimize a convex function")

        # a simple function with redundant inputs and tol instead of parameter tolerance
        p = optimize(loss=lambda p: (p[0]-2)**2+(p[1]-1)**4, max_vals=[5, 5, 5], min_vals=[0, 0, 5], deviation_tol=1.E-6, divide_range="shrinking")
        self.assertAlmostEqual(p[0], 2, places=3, msg="Optimizer should easily optimize a convex function")
        self.assertAlmostEqual(p[1], 1, places=3, msg="Optimizer should easily optimize a convex function")

        # https://en.wikipedia.org/wiki/Test_functions_for_optimization

        # Beale function
        p = optimize(loss=lambda p: (1.5-p[0]+p[0]*p[1])**2+(2.25-p[0]+p[0]*p[1]**2)**2+(2.625-p[0]+p[0]*p[1]**3)**2, max_vals=[4.5, 4.5], min_vals=[-4.5, -4.5], parameter_tol=1.E-8)
        self.assertAlmostEqual(p[0], 3, places=6, msg="Optimizer should optimize the Beale function")
        self.assertAlmostEqual(p[1], 0.5, places=6, msg="Optimizer should optimize the Beale function")

        # Booth function
        p = optimize(loss=lambda p: (p[0]+2*p[1]-7)**2+(2*p[0]+p[1]-5)**2, max_vals=[10, 10], min_vals=[-10, -10], parameter_tol=1.E-8)
        self.assertAlmostEqual(p[0], 1, places=6, msg="Optimizer should optimize the Booth function")
        self.assertAlmostEqual(p[1], 3, places=6, msg="Optimizer should optimize the Booth function")

        # Beale function with depth instead of small divide range
        p = optimize(loss=lambda p: (1.5-p[0]+p[0]*p[1])**2+(2.25-p[0]+p[0]*p[1]**2)**2+(2.625-p[0]+p[0]*p[1]**3)**2,
                     max_vals=[4.5, 4.5], min_vals=[-4.5, -4.5], parameter_tol=1.E-8, divide_range=2, depth=100)
        self.assertAlmostEqual(p[0], 3, places=6, msg="Optimizer should optimize the Beale function")
        self.assertAlmostEqual(p[1], 0.5, places=6, msg="Optimizer should optimize the Beale function")

    def test_fairness(self):
        from pygrank import Normalize, FairPersonalizer, Mabs, PageRank, pRule
        H = PageRank(assume_immutability=True, normalization="symmetric")
        algorithms = {
            "FairPers": lambda G, p, s: Normalize(FairPersonalizer(H, error_type=Mabs, max_residual=0)).rank(G, p, sensitive=s),
            "FairPers-C": lambda G, p, s: Normalize(FairPersonalizer(H, .80, pRule_weight=10, error_type=Mabs, max_residual=0)).rank(G, p, sensitive=s),
            "FairPersKL": lambda G, p, s: Normalize(FairPersonalizer(H, max_residual=0)).rank(G, p, sensitive=s),
            "FairPersKL-C": lambda G, p, s: Normalize(FairPersonalizer(H, .80, pRule_weight=10, max_residual=0)).rank(G, p, sensitive=s),
        }
        graph, sensitive, labels = fairness_dataset("facebook0", 0, sensitive_group=1, path="data/")
        for algorithm in algorithms.values():
            ranks = algorithm(graph, labels, sensitive)
            self.assertGreater(pRule(sensitive)(ranks), 0.8, "should satisfy fairness requirements")


