import unittest
import networkx as nx
from .example_graph import test_graph


class Test(unittest.TestCase):
    def test_pagerank(self):
        from pygrank.algorithms.pagerank import PageRank
        G = test_graph()
        test_result = PageRank(normalization='col').rank(G)
        nx_result = nx.pagerank_scipy(G)
        abs_diffs = sum(abs(test_result[v] - nx_result[v]) for v in nx_result.keys()) / len(nx_result)
        self.assertAlmostEqual(abs_diffs, 0, places=16, msg="PageRank compliance with nx results")

    """
    def test_absorbing_walk(self):
        from pygrank.algorithms.pagerank import PageRank
        from pygrank.algorithms.pagerank import AbsorbingRank
        G = test_graph()
        personalization = {"A": 1, "B": 1}
        pagerank_result = PageRank(normalization='col').rank(G, personalization)
        absorbing_result = AbsorbingRank(normalization='col', max_iters=1000).rank(G, personalization, absorption={v: G.degree(v) for v in G})
        abs_diffs = sum(abs(pagerank_result[v] - absorbing_result[v]) for v in pagerank_result.keys()) / len(pagerank_result)
        self.assertAlmostEqual(abs_diffs, 0, places=16, msg="Absorbing Random Walks compliance with PageRank results")
    """

    def test_heat_kernel_locality(self):
        from pygrank.algorithms.pagerank import PageRank
        from pygrank.algorithms.pagerank import HeatKernel
        G = test_graph()
        personalization = {"A": 1, "B": 1}
        pagerank = PageRank().rank(G, personalization)
        heatkernel = HeatKernel().rank(G, personalization)
        self.assertLess(pagerank['A']/sum(pagerank.values()), heatkernel['A']/sum(heatkernel.values()), msg="HeatKernel more local than PageRank")
        self.assertLess(heatkernel['I']/sum(heatkernel.values()), pagerank['I']/sum(pagerank.values()), msg="HeatKernel more local than PageRank")

    def test_venuerank(self):
        from pygrank.algorithms.pagerank import PageRank
        from pygrank.algorithms.postprocess import Ordinals
        from scipy.stats import spearmanr
        G = nx.fast_gnp_random_graph(600, 0.001, seed=1)
        ranker1 = PageRank(max_iters=10000, converge_to_eigenvectors=True, tol=1.E-12)
        ranks1 = ranker1.rank(G, personalization={0: 1, 1: 1})
        ranker2 = PageRank(alpha=0.99, max_iters=10000, tol=1.E-12)
        ranks2 = ranker2.rank(G, personalization={0: 1, 1: 1})
        self.assertLess(ranker1.convergence.iteration, ranker2.convergence.iteration / 10,
                        msg="converge_to_eigenvectors (VenueRank) should be much faster in difficult-to-rank graphs")
        corr = spearmanr(list(Ordinals().transform(ranks1).values()), list(Ordinals().transform(ranks2).values()))
        self.assertAlmostEqual(corr[0], 1., 4, msg="converge_to_eigenvectors (VenueRank) should yield similar order to the one of small restart probability")


