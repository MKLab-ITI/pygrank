import unittest
import networkx as nx
from tests.example_graph import test_graph


class Test(unittest.TestCase):
    def test_normalize_range(self):
        from pygrank.algorithms.adhoc import PageRank
        from pygrank.algorithms.postprocess import Normalize
        G = test_graph()
        r = Normalize(PageRank(), "range").rank(G)
        self.assertAlmostEqual(min(r[v] for v in G), 0, places=16)
        self.assertAlmostEqual(max(r[v] for v in G), 1, places=16)

    def test_normalize_sum(self):
        from pygrank.algorithms.adhoc import PageRank
        from pygrank.algorithms.postprocess import Normalize
        G = test_graph()
        r = Normalize(PageRank(), "sum").rank(G)
        self.assertAlmostEqual(sum(r[v] for v in G), 1, places=16)

    def test_normalize_invalid(self):
        from pygrank.algorithms.adhoc import PageRank
        from pygrank.algorithms.postprocess import Normalize
        G = test_graph()
        with self.assertRaises(Exception):
            Normalize(PageRank(), "unknown").rank(G)

    def test_ordinals(self):
        from pygrank.algorithms.adhoc import PageRank
        from pygrank.algorithms.postprocess import Ordinals
        G = test_graph()
        test_result = Ordinals(Ordinals(Ordinals(PageRank(normalization='col')))).rank(G, {"A": 1}) # three ordinal transformations are the same as one
        self.assertAlmostEqual(test_result["A"], 1, places=16, msg="Ordinals should compute without errors (seed node is highest rank in small graphs)")

    def test_normalization(self):
        from pygrank.algorithms.adhoc import PageRank
        from pygrank.algorithms.postprocess import Normalize
        G = test_graph()
        test_result = Normalize("sum", PageRank(normalization='col')).rank(G)
        self.assertAlmostEqual(sum(test_result.values()), 1, places=16, msg="Sum normalization should sum to 1")
        test_result = Normalize("max", PageRank(normalization='col')).rank(G)
        self.assertAlmostEqual(max(test_result.values()), 1, places=16, msg="Max should have max 1")
        test_result = Normalize("sum").transform(PageRank(normalization='col').rank(G))
        self.assertAlmostEqual(sum(test_result.values()), 1, places=16, msg="Normalization should be able to use transformations")

    def test_oversampling(self):
        from pygrank.algorithms.adhoc import PageRank
        from pygrank.algorithms.postprocess.oversampling import SeedOversampling
        G = test_graph()
        test_result = SeedOversampling(PageRank(normalization='col')).rank(G, {"A": 1})

    def test_abstract_postprocessor(self):
        from pygrank.algorithms.adhoc import PageRank
        from pygrank.algorithms.postprocess import Postprocessor
        with self.assertRaises(Exception):
            p = Postprocessor(PageRank())
            G = test_graph()
            p.rank(G)

    def test_optimizer_errors(self):
        from pygrank.algorithms.utils import optimize
        with self.assertRaises(Exception):
            optimize(loss=lambda p: (p[0]-2)**2+(p[1]-1)**4, max_vals=[5, 5], parameter_tol=1.E-8, divide_range=1)
        with self.assertRaises(Exception):
            optimize(loss=lambda p: (p[0]-2)**2+(p[1]-1)**4, max_vals=[5, 5], min_vals=[5, 6])

    def test_optimizer(self):
        from pygrank.algorithms.utils import optimize

        # a simple function
        p = optimize(loss=lambda p: (p[0]-2)**2+(p[1]-1)**4, max_vals=[5, 5], parameter_tol=1.E-8)
        self.assertAlmostEqual(p[0], 2, places=6, msg="Optimizer should easily optimize a convex function")
        self.assertAlmostEqual(p[1], 1, places=6, msg="Optimizer should easily optimize a convex function")

        # a simple function with redundant inputs and tol instead of parameter tolerance
        p = optimize(loss=lambda p: (p[0]-2)**2+(p[1]-1)**4, max_vals=[5, 5, 5], min_vals=[0, 0, 5], tol=1.E-6, divide_range="shrinking")
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


