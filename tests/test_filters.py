import unittest
import networkx as nx
from tests.example_graph import test_graph, test_block_model_graph


class Test(unittest.TestCase):

    def test_backend_conversion(self):
        from pygrank import backend
        self.assertEqual(backend.sum(backend.to_array([1, 2, 3])), 6)
        self.assertEqual(backend.sum(backend.dot(backend.exp(backend.log(backend.to_array([4, 5]))), backend.to_array([2, 2]))), 18)
        primitive = backend.to_array([1, 2, 3])
        self.assertTrue(id(primitive) == id(backend.to_array(primitive, copy_array=False)))
        self.assertFalse(id(primitive) == id(backend.to_array(primitive, copy_array=True)))

    def test_callers(self):
        from pygrank import call, remove_used_args

        def test_func(x, y=0):
            return x+y

        self.assertEqual(call(test_func, {"x":1, "y": 2, "z": 3}), 3)
        self.assertEqual(call(test_func, {"y": 2, "z": 3}, [1]), 3)
        self.assertEqual(len(remove_used_args(test_func, {"y": 2, "z": 3}, [1])), 1)
        with self.assertRaises(Exception):
            call(test_func, {"y": 2, "z": 3}, [1, 2])

    def test_signal(self):
        from pygrank import GraphSignal
        with self.assertRaises(Exception):
            GraphSignal([1, 2, 3], [1, 2])
        signal = GraphSignal(test_graph(), {"A": 1, "B": 2})
        del signal["A"]
        self.assertEqual(signal["A"], 0)
        self.assertEqual(signal["B"], 2)

    def test_unused_arguments(self):
        from pygrank import PageRank
        with self.assertRaises(Exception):
            PageRank(krylov_dims=5)

    def test_node_ranking(self):
        from pygrank import NodeRanking, PageRank
        G = test_graph()
        with self.assertRaises(Exception):
            NodeRanking().rank(G)
        ranker = PageRank(normalization='col', tol=1.E-9)
        test_result = ranker.rank(G)
        test_result2 = ranker(G)
        abs_diffs = sum(abs(test_result[v] - test_result2[v]) for v in test_result2.keys()) / len(test_result2)
        self.assertAlmostEqual(abs_diffs, 0, places=16, msg="PageRank compliance with nx results")

    def test_abstract_filter(self):
        from pygrank.algorithms.filters.abstract_filters import GraphFilter, RecursiveGraphFilter, ClosedFormGraphFilter
        G = test_graph()
        with self.assertRaises(Exception):
            GraphFilter().rank(G, {})
        with self.assertRaises(Exception):
            GraphFilter().rank(G)
        with self.assertRaises(Exception):
            RecursiveGraphFilter().rank(G)
        with self.assertRaises(Exception):
            ClosedFormGraphFilter().rank(G)

    def test_prevent_passing_node_lists_as_graphs(self):
        from pygrank import PageRank
        with self.assertRaises(Exception):
            PageRank().rank(list(test_graph()))

    def test_non_convergence(self):
        from pygrank import PageRank
        with self.assertRaises(Exception):
            PageRank(max_iters=5).rank(test_graph())

    def test_custom_runs(self):
        from pygrank import PageRank, GenericGraphFilter, Mabs, Normalize
        G = test_graph()
        algorithmn = PageRank(0.85, max_iters=5, error_type="iters")
        ranks1 = algorithmn.rank(G, {"A": 1})
        self.assertTrue("6" in str(algorithmn.convergence))
        # TODO find why the following is not exactly the same
        """ ranks2 = Normalize(GenericGraphFilter([0.85]*1, tol=1.E-12)).rank(G, {"A": 1})
        print(ranks1.np-ranks2.np)
        self.assertEqual(Mabs(ranks1)(ranks2), 0)
        """

    def test_completion(self):
        from pygrank import PageRank, HeatKernel, AbsorbingWalks
        G = test_graph()
        PageRank().rank(G)
        HeatKernel().rank(G)
        AbsorbingWalks().rank(G)

    def test_pagerank(self):
        from pygrank import PageRank
        G = test_graph()
        test_result = PageRank(normalization='col', tol=1.E-9).rank(G)
        nx_result = nx.pagerank(G, tol=1.E-9)
        abs_diffs = sum(abs(test_result[v] - nx_result[v]) for v in nx_result.keys()) / len(nx_result)
        self.assertAlmostEqual(abs_diffs, 0, places=12, msg="PageRank compliance with nx results")

    def test_quotient(self):
        from pygrank import PageRank, Normalize
        G = test_graph()
        test_result = PageRank(normalization='symmetric', tol=1.E-9, use_quotient=True).rank(G)
        norm_result = PageRank(normalization='symmetric', tol=1.E-9, use_quotient=Normalize("sum")).rank(G)
        abs_diffs = sum(abs(test_result[v] - norm_result[v]) for v in norm_result.keys()) / len(norm_result)
        self.assertAlmostEqual(abs_diffs, 0, places=12, msg="Using quotient yields the same result")

    def test_oversampling(self):
        from pygrank import PageRank, to_signal, SeedOversampling, BoostedSeedOversampling, split, NDCG
        import random
        G, groups = test_block_model_graph(nodes=600, seed=1)
        group = groups[0]
        random.seed(1)
        training, evaluation = split(list(group), training_samples=3)
        training, evaluation = to_signal(G, {v: 1 for v in training}), to_signal(G, {v: 1 for v in evaluation})

        base_result = NDCG(evaluation, training).evaluate(PageRank(0.99).rank(G, training))
        so_result = NDCG(evaluation, training).evaluate(SeedOversampling(PageRank(0.99)).rank(G, training))
        bso_result = NDCG(evaluation, training).evaluate(BoostedSeedOversampling(PageRank(0.99)).rank(G, training))
        self.assertLessEqual(base_result, so_result)
        self.assertLessEqual(so_result, bso_result)

        SeedOversampling(PageRank(0.99), "top").rank(G, training)

    def test_implicit_graph(self):
        from pygrank.algorithms import PageRank
        from pygrank.algorithms import to_signal
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
        from pygrank.algorithms import HeatKernel
        from pygrank.algorithms import Normalize
        G = test_graph()
        result = Normalize(HeatKernel(normalization='symmetric')).rank(G)
        result_lanczos = Normalize(HeatKernel(normalization='symmetric', krylov_dims=5)).rank(G, personalization=None)
        abs_diffs = sum(abs(result[v] - result_lanczos[v]) for v in result_lanczos.keys()) / len(result_lanczos)
        self.assertAlmostEqual(abs_diffs, 0, places=0, msg="Krylov decomposition yields small error")

        with self.assertRaises(Exception):
            HeatKernel(normalization='col', krylov_dims=5).rank(G)

    def test_absorbing_walk(self):
        from pygrank.algorithms import PageRank
        from pygrank.algorithms import AbsorbingWalks
        G = test_graph()
        personalization = {"A": 1, "B": 1}
        pagerank_result = PageRank(normalization='col').rank(G, personalization)
        absorbing_result = AbsorbingWalks(0.85, normalization='col', max_iters=1000).rank(G, personalization)
        abs_diffs = sum(abs(pagerank_result[v] - absorbing_result[v]) for v in pagerank_result.keys()) / len(
            pagerank_result)
        self.assertAlmostEqual(abs_diffs, 0, places=16, msg="Absorbing Random Walks compliance with PageRank results")

    def test_heat_kernel_locality(self):
        from pygrank.algorithms import PageRank
        from pygrank.algorithms import HeatKernel
        G = test_graph()
        personalization = {"A": 1, "B": 1}
        pagerank = PageRank().rank(G, personalization)
        heatkernel = HeatKernel().rank(G, personalization)
        self.assertLess(pagerank['A'] / sum(pagerank.values()), heatkernel['A'] / sum(heatkernel.values()),
                        msg="HeatKernel should be more local than PageRank")
        self.assertLess(heatkernel['I'] / sum(heatkernel.values()), pagerank['I'] / sum(pagerank.values()),
                        msg="HeatKernel should be more local than PageRank")

    def test_biased_kernel_locality(self):
        from pygrank.algorithms import PageRank
        from pygrank.algorithms import BiasedKernel
        G = test_graph()
        personalization = {"A": 1, "B": 1}
        pagerank = PageRank().rank(G, personalization)
        heatkernel = BiasedKernel().rank(G, personalization)
        self.assertLess(pagerank['A'] / sum(pagerank.values()), heatkernel['A'] / sum(heatkernel.values()),
                        msg="BiasedRank should be more local than PageRank")
        self.assertLess(heatkernel['I'] / sum(heatkernel.values()), pagerank['I'] / sum(pagerank.values()),
                        msg="BiasedRank should be more local than PageRank")

    def test_venuerank(self):
        from pygrank.algorithms import PageRank
        from pygrank.algorithms import Ordinals
        from scipy.stats import spearmanr
        G = nx.fast_gnp_random_graph(600, 0.001, seed=1)
        ranker1 = PageRank(max_iters=10000, converge_to_eigenvectors=True, tol=1.E-12)
        ranks1 = ranker1.rank(G, personalization={0: 1, 1: 1})
        ranker2 = PageRank(alpha=0.99, max_iters=10000, tol=1.E-12)
        ranks2 = ranker2.rank(G, personalization={0: 1, 1: 1})
        self.assertLess(ranker1.convergence.iteration, ranker2.convergence.iteration / 10,
                        msg="converge_to_eigenvectors should be much faster in difficult-to-rank graphs")
        corr = spearmanr(list(Ordinals().transform(ranks1).values()), list(Ordinals().transform(ranks2).values()))
        self.assertAlmostEqual(corr[0], 1., 4,
                               msg="converge_to_eigenvectors should yield similar order to small restart probability")

    def test_learnable(self):
        from pygrank.algorithms import optimize
        from pygrank.algorithms import GenericGraphFilter, HeatKernel
        from pygrank.algorithms import to_signal, preprocessor
        from pygrank.measures.utils import split
        from pygrank.measures import AUC
        from pygrank.algorithms import Normalize
        import random
        G, groups = test_block_model_graph(nodes=600, seed=1)
        group = groups[0]
        random.seed(1)
        used_for_training, evaluation = split(list(group), training_samples=0.5)
        training, validation = split(used_for_training, training_samples=0.5)
        training = to_signal(G, {v: 1 for v in training})
        validation = to_signal(G, {v: 1 for v in validation})
        evaluation = to_signal(G, {v: 1 for v in evaluation})

        pre = preprocessor("symmetric", True)
        optimal_params = optimize(lambda params: -AUC(validation, exclude=training).evaluate(
            Normalize("sum", GenericGraphFilter(params, to_scipy=pre, max_iters=10000)).rank(G, training)),
                          max_vals=[1] * 5, deviation_tol=0.01, divide_range=2, verbose=True, partitions=5)
        learnable_result = AUC(validation, exclude=evaluation).evaluate(
            GenericGraphFilter(optimal_params, to_scipy=pre, max_iters=10000).rank(G, training))
        heat_kernel_result = AUC(evaluation, used_for_training).evaluate(
            HeatKernel(7, to_scipy=pre, max_iters=10000).rank(G, training))
        self.assertLess(heat_kernel_result, learnable_result,
                        msg="Learnable parameters should be meaningful")
        self.assertGreater(heat_kernel_result,
                           AUC(evaluation).evaluate(HeatKernel(7, to_scipy=pre, max_iters=10000).rank(G, training)),
                           msg="Metrics correctly apply exclude filter to not skew results")

    def test_chebyshev(self):
        from pygrank import optimize, GenericGraphFilter, HeatKernel, to_signal, preprocessor, split, AUC, Normalize
        import random
        G, groups = test_block_model_graph(nodes=600, seed=1)
        group = groups[0]
        random.seed(1)
        used_for_training, evaluation = split(to_signal(G, {v: 1 for v in group}), training_samples=0.5)
        training, validation = split(used_for_training, training_samples=0.5)

        pre = preprocessor("symmetric", True)
        with self.assertRaises(Exception):
            optimize(lambda params: -AUC(validation, exclude=training).evaluate(
                Normalize("sum",
                          GenericGraphFilter(params, coefficient_type="unknown", to_scipy=pre, max_iters=10000)).rank(G, training)),
                     max_vals=[1] * 5, min_vals=[0] * 5, deviation_tol=0.01, divide_range=2, verbose=True, partitions=5)

        optimal_params = optimize(lambda params: -AUC(validation, exclude=training).evaluate(
            Normalize("sum",
                      GenericGraphFilter(params, coefficient_type="Chebyshev", to_scipy=pre, max_iters=10000)).rank(G, training)),
                                  max_vals=[1] * 5, min_vals=[0] * 5, deviation_tol=0.01, divide_range=2, verbose=True,
                                  partitions=5)
        learnable_result = AUC(validation, exclude=evaluation).evaluate(
            GenericGraphFilter(optimal_params, to_scipy=pre, max_iters=10000).rank(G, training))
        heat_kernel_result = AUC(evaluation, exclude=used_for_training).evaluate(
            HeatKernel(7, to_scipy=pre, max_iters=10000).rank(G, training))
        self.assertLess(heat_kernel_result, learnable_result,
                        msg="Learnable parameters should be meaningful")
        self.assertGreater(heat_kernel_result,
                           AUC(evaluation).evaluate(HeatKernel(7, to_scipy=pre, max_iters=10000).rank(G, training)),
                           msg="Metrics correctly apply exclude filter to not skew results")

    def test_preprocessor(self):
        from pygrank import preprocessor, MethodHasher
        G = test_graph()
        with self.assertRaises(Exception):
            pre = preprocessor(normalization="unknown", assume_immutability=True)
            pre(G)

        pre = preprocessor(normalization="col", assume_immutability=False)
        G = test_graph()
        res1 = pre(G)
        res2 = pre(G)
        self.assertTrue(id(res1) != id(res2),
                        msg="When immutability is not assumed, different objects are returned")

        pre = MethodHasher(preprocessor, assume_immutability=True)
        G = test_graph()
        res1 = pre(G)
        pre.assume_immutability = False  # have the ability to switch immutability off midway
        res2 = pre(G)
        self.assertTrue(id(res1) != id(res2),
                        msg="When immutability is not assumed, different objects are returned")

        pre = preprocessor(normalization="col", assume_immutability=True)
        G = test_graph()
        res1 = pre(G)
        pre.clear_hashed()
        res2 = pre(G)
        self.assertTrue(id(res1) != id(res2),
                        msg="When immutability is assumed but data cleared, different objects are returned")

    def test_backend(self):
        from pygrank import backend
        backend.load_backend("tensorflow")
        self.assertEqual(backend.backend_name(), "tensorflow")
        self.test_backend_conversion()
        self.test_pagerank()
        self.test_venuerank()
        self.test_absorbing_walk()
        self.test_prevent_passing_node_lists_as_graphs()
        backend.load_backend("numpy")
        self.assertEqual(backend.backend_name(), "numpy")
        with self.assertRaises(Exception):
            backend.load_backend("unknown")
        self.assertEqual(backend.backend_name(), "numpy")
        self.test_pagerank()
