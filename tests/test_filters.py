import unittest
import networkx as nx
from .example_graph import test_graph, test_block_model_graph

# coverage run --source=pygrank -m unittest tests/test_filters.py
# coverage html

class Test(unittest.TestCase):
    def test_completion(self):
        from pygrank.algorithms.adhoc import PageRank, HeatKernel, AbsorbingWalks
        G = test_graph()
        PageRank().rank(G)
        HeatKernel().rank(G)
        AbsorbingWalks().rank(G)

    def test_pagerank(self):
        from pygrank.algorithms.adhoc import PageRank
        G = test_graph()
        test_result = PageRank(normalization='col').rank(G)
        nx_result = nx.pagerank_scipy(G)
        abs_diffs = sum(abs(test_result[v] - nx_result[v]) for v in nx_result.keys()) / len(nx_result)
        self.assertAlmostEqual(abs_diffs, 0, places=16, msg="PageRank compliance with nx results")

    def test_implicity_graph(self):
        from pygrank.algorithms.adhoc import PageRank
        from pygrank.algorithms.utils import to_signal
        G = test_graph()
        signal = to_signal(G, {"A": 1})
        test_result1 = PageRank(normalization='col').rank(signal, signal)
        test_result2 = PageRank(normalization='col').rank(personalization=signal)
        abs_diffs = sum(abs(test_result1[v] - test_result2[v]) for v in test_result2.keys()) / len(test_result2)
        self.assertAlmostEqual(abs_diffs, 0, places=16, msg="Same result if graph not used")

        with self.assertRaises(Exception):
            PageRank(normalization='col').rank(personalization={"A": 1})

        with self.assertRaises(Exception):
            PageRank(normalization='col').rank(test_graph(), signal)


    def test_lanczos(self):
        from pygrank.algorithms.adhoc import HeatKernel
        from pygrank.algorithms.postprocess import Normalize
        G = test_graph()
        test_result = Normalize(HeatKernel(normalization='symmetric')).rank(G)
        test_result_lanczos = Normalize(HeatKernel(normalization='symmetric', krylov_dims=5)).rank(G)
        abs_diffs = sum(abs(test_result[v] - test_result_lanczos[v]) for v in test_result_lanczos.keys()) / len(test_result_lanczos)
        self.assertAlmostEqual(abs_diffs, 0, places=0, msg="Krylov decomposition yields small error")

    def test_absorbing_walk(self):
        from pygrank.algorithms.adhoc import PageRank
        from pygrank.algorithms.adhoc import AbsorbingWalks
        G = test_graph()
        personalization = {"A": 1, "B": 1}
        pagerank_result = PageRank(normalization='col').rank(G, personalization)
        absorbing_result = AbsorbingWalks(0.85, normalization='col', max_iters=1000).rank(G, personalization)#, absorption={v: G.degree(v) for v in G})
        abs_diffs = sum(abs(pagerank_result[v] - absorbing_result[v]) for v in pagerank_result.keys()) / len(pagerank_result)
        self.assertAlmostEqual(abs_diffs, 0, places=16, msg="Absorbing Random Walks compliance with PageRank results")

    def test_heat_kernel_locality(self):
        from pygrank.algorithms.adhoc import PageRank
        from pygrank.algorithms.adhoc import HeatKernel
        G = test_graph()
        personalization = {"A": 1, "B": 1}
        pagerank = PageRank().rank(G, personalization)
        heatkernel = HeatKernel().rank(G, personalization)
        #self.assertLess(pagerank['A']/sum(pagerank.values()), heatkernel['A']/sum(heatkernel.values()), msg="HeatKernel more local than PageRank")
        #self.assertLess(heatkernel['I']/sum(heatkernel.values()), pagerank['I']/sum(pagerank.values()), msg="HeatKernel more local than PageRank")

    def test_venuerank(self):
        from pygrank.algorithms.adhoc import PageRank
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

    def test_learnable(self):
        from pygrank.algorithms.utils.optimization import optimize
        from pygrank.algorithms import GenericGraphFilter, HeatKernel
        from pygrank.algorithms.utils import to_signal, preprocessor
        from pygrank.measures.utils import split_groups
        from pygrank.measures import AUC
        from pygrank.algorithms.postprocess import Normalize
        import random
        G, groups = test_block_model_graph(nodes=600, seed=1)
        group = groups[0]
        random.seed(1)
        used_for_training, evaluation = split_groups(list(group), training_samples=0.5)
        training, validation = split_groups(used_for_training, training_samples=0.5)
        training, validation, evaluation = to_signal(G, {v: 1 for v in training}), to_signal(G, {v: 1 for v in validation}), to_signal(G, {v: 1 for v in evaluation})

        pre = preprocessor("symmetric", True)
        params = optimize(lambda params: -AUC(validation, exclude=evaluation).evaluate(
            Normalize("sum", GenericGraphFilter(params, to_scipy=pre, max_iters=10000)).rank(G, training)),
                          max_vals=[1]*5, tol=0.01, divide_range=2, verbose=True, partitions=5)
        learnable_result = AUC(validation, exclude=evaluation).evaluate(
            GenericGraphFilter(params, to_scipy=pre, max_iters=10000).rank(G, training))
        heat_kernel_result = AUC(evaluation, exclude=used_for_training).evaluate(HeatKernel(7, to_scipy=pre, max_iters=10000).rank(G, training))
        self.assertLess(heat_kernel_result, learnable_result, msg="Learnable parameters should be meaningful")
        self.assertGreater(heat_kernel_result, AUC(evaluation).evaluate(HeatKernel(7, to_scipy=pre, max_iters=10000).rank(G, training)),
                        msg="Metrics correctly apply exclude filter to not skew results")

    def test_preprocessor(self):
        from pygrank.algorithms.utils import preprocessor
        G = test_graph()
        with self.assertRaises(Exception):
            pre = preprocessor(normalization="unknown", assume_immutability=True)
            pre(G)

        pre = preprocessor(normalization="col", assume_immutability=False)
        G = test_graph()
        res1 = pre(G)
        res2 = pre(G)
        self.assertTrue(id(res1) != id(res2), msg="When immutability is not assumed, different objects are returned")

        pre = preprocessor(normalization="col", assume_immutability=True)
        G = test_graph()
        res1 = pre(G)
        pre.clear_hashed()
        res2 = pre(G)
        self.assertTrue(id(res1) != id(res2), msg="When immutability is assumed but data cleared, different objects are returned")
