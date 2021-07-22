import unittest
import networkx as nx
from .example_graph import test_graph


class Test(unittest.TestCase):
    def test_ordinals(self):
        from pygrank.algorithms.pagerank import PageRank
        from pygrank.algorithms.postprocess import Ordinals
        G = test_graph()
        test_result = Ordinals(Ordinals(Ordinals(PageRank(normalization='col')))).rank(G, {"A": 1}) # three ordinal transformations are the same as one
        self.assertAlmostEqual(test_result["A"], 1, places=16, msg="Ordinals should compute without errors (seed node is highest rank in small graphs)")

    def test_normalization(self):
        from pygrank.algorithms.pagerank import PageRank
        from pygrank.algorithms.postprocess import Normalize
        G = test_graph()
        test_result = Normalize("sum", PageRank(normalization='col')).rank(G)
        self.assertAlmostEqual(sum(test_result.values()), 1, places=16, msg="Sum normalization should sum to 1")
        test_result = Normalize("max", PageRank(normalization='col')).rank(G)
        self.assertAlmostEqual(max(test_result.values()), 1, places=16, msg="Max should have max 1")
        test_result = Normalize("sum").transform(PageRank(normalization='col').rank(G))
        self.assertAlmostEqual(sum(test_result.values()), 1, places=16, msg="Normalization should be able to use transformations")

    def test_oversampling(self):
        from pygrank.algorithms.pagerank import PageRank
        from pygrank.algorithms.oversampling import SeedOversampling
        G = test_graph()
        test_result = SeedOversampling(PageRank(normalization='col')).rank(G, {"A": 1})

