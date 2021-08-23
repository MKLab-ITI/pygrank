import unittest
import networkx as nx
import pygrank as pg


class Test(unittest.TestCase):

    def test_backend_conversion(self):
        self.assertEqual(pg.sum(pg.to_array([1, 2, 3])), 6)
        self.assertEqual(pg.sum(pg.dot(pg.exp(pg.log(pg.to_array([4, 5]))), pg.to_array([2, 2]))), 18)
        primitive = pg.to_array([1, 2, 3])
        self.assertTrue(id(primitive) == id(pg.to_array(primitive, copy_array=False)))
        self.assertFalse(id(primitive) == id(pg.to_array(primitive, copy_array=True)))

    def test_callers(self):
        def test_func(x, y=0):
            return x+y

        self.assertEqual(pg.call(test_func, {"x":1, "y": 2, "z": 3}), 3)
        self.assertEqual(pg.call(test_func, {"y": 2, "z": 3}, [1]), 3)
        self.assertEqual(len(pg.remove_used_args(test_func, {"y": 2, "z": 3}, [1])), 1)
        with self.assertRaises(Exception):
            pg.call(test_func, {"y": 2, "z": 3}, [1, 2])
        with self.assertRaises(Exception):
            pg.PageRank(krylov_dims=5)

    def test_signal(self):
        with self.assertRaises(Exception):
            pg.GraphSignal([1, 2, 3], [1, 2])
        signal = pg.GraphSignal(next(pg.load_datasets_graph(["graph9"])), {"A": 1, "B": 2})
        del signal["A"]
        self.assertEqual(signal["A"], 0)
        self.assertEqual(signal["B"], 2)

    def test_zero_personalization(self):
        self.assertEqual(pg.sum(pg.PageRank()(next(pg.load_datasets_graph(["graph9"])),{}).np), 0)

    def test_node_ranking(self):
        from pygrank import NodeRanking, PageRank
        graph = next(pg.load_datasets_graph(["graph9"]))
        with self.assertRaises(Exception):
            NodeRanking().rank(graph)
        ranker = PageRank(normalization='col', tol=1.E-9)
        test_result = ranker.rank(graph)
        test_result2 = ranker(graph)
        abs_diffs = sum(abs(test_result[v] - test_result2[v]) for v in test_result2.keys()) / len(test_result2)
        self.assertAlmostEqual(abs_diffs, 0, places=16, msg="PageRank compliance with nx results")

    def test_abstract_filter(self):
        graph = next(pg.load_datasets_graph(["graph5"]))
        with self.assertRaises(Exception):
            pg.GraphFilter().rank(graph)
        with self.assertRaises(Exception):
            pg.RecursiveGraphFilter().rank(graph)
        with self.assertRaises(Exception):
            pg.ClosedFormGraphFilter().rank(graph)
        with self.assertRaises(Exception):
            pg.Tuner().rank(graph)

    def test_prevent_passing_node_lists_as_graphs(self):
        graph = next(pg.load_datasets_graph(["graph5"]))
        with self.assertRaises(Exception):
            pg.PageRank().rank(list(graph))

    def test_non_convergence(self):
        graph = next(pg.load_datasets_graph(["graph9"]))
        with self.assertRaises(Exception):
            pg.PageRank(max_iters=5).rank(graph)

    def test_custom_runs(self):
        graph = next(pg.load_datasets_graph(["graph9"]))
        ranks1 = pg.Normalize(pg.PageRank(0.85, tol=1.E-12, max_iters=1000)).rank(graph, {"A": 1})
        # TODO find why the following is not exactly the same
        ranks2 = pg.Normalize(pg.GenericGraphFilter([0.85**i for i in range(20)], tol=1.E-12)).rank(graph, {"A": 1})
        #print(ranks1.np-ranks2.np)
        #self.assertAlmostEqual(pg.Mabs(ranks1)(ranks2), 0, places=11)

    def test_completion(self):
        G = test_graph()
        pg.PageRank().rank(G)
        pg.HeatKernel().rank(G)
        pg.AbsorbingWalks().rank(G)

    def test_pagerank(self):
        G = test_graph()
        test_result = pg.PageRank(normalization='col', tol=1.E-9, personalization_transform=pg.Normalize("sum")).rank(G)
        nx_result = nx.pagerank(G, tol=1.E-9)
        abs_diffs = sum(abs(test_result[v] - nx_result[v]) for v in nx_result.keys()) / len(nx_result)
        self.assertAlmostEqual(abs_diffs, 0, places=12, msg="PageRank compliance with nx results")

    def test_quotient(self):
        G = test_graph()
        test_result = pg.PageRank(normalization='symmetric', tol=1.E-9, use_quotient=True).rank(G)
        norm_result = pg.PageRank(normalization='symmetric', tol=1.E-9, use_quotient=pg.Normalize("sum")).rank(G)
        abs_diffs = sum(abs(test_result[v] - norm_result[v]) for v in norm_result.keys()) / len(norm_result)
        self.assertAlmostEqual(abs_diffs, 0, places=12, msg="Using quotient yields the same result")

    def test_oversampling(self):
        import random
        G, groups = test_block_model_graph(nodes=600, seed=1)
        group = groups[0]
        random.seed(1)
        training, evaluation = pg.split(list(group), training_samples=3)
        training, evaluation = pg.to_signal(G, {v: 1 for v in training}), pg.to_signal(G, {v: 1 for v in evaluation})

        base_result = pg.NDCG(evaluation, training).evaluate(pg.PageRank(0.99).rank(G, training))
        so_result = pg.NDCG(evaluation, training).evaluate(pg.SeedOversampling(pg.PageRank(0.99)).rank(G, training))
        bso_result = pg.NDCG(evaluation, training).evaluate(pg.BoostedSeedOversampling(pg.PageRank(0.99)).rank(G, training))
        self.assertLessEqual(base_result, so_result)
        self.assertLessEqual(so_result, bso_result)

        pg.SeedOversampling(pg.PageRank(0.99), "top").rank(G, training)

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
        G = nx.fast_gnp_random_graph(600, 0.001, seed=1)
        ranker1 = pg.PageRank(max_iters=10000, converge_to_eigenvectors=True, tol=1.E-12)
        ranks1 = ranker1.rank(G, personalization={0: 1, 1: 1})
        ranker2 = pg.PageRank(alpha=0.99, max_iters=10000, tol=1.E-12)
        ranks2 = ranker2.rank(G, personalization={0: 1, 1: 1})
        self.assertLess(ranker1.convergence.iteration, ranker2.convergence.iteration / 10,
                        msg="converge_to_eigenvectors should be much faster in difficult-to-rank graphs")
        corr = pg.SpearmanCorrelation(pg.Ordinals().transform(ranks1))(pg.Ordinals().transform(ranks2))
        self.assertAlmostEqual(corr, 1., 4,
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
            Normalize("sum", GenericGraphFilter(params, preprocessor=pre, max_iters=10000)).rank(G, training)),
                          max_vals=[1] * 5, deviation_tol=0.01, divide_range=2, verbose=False, partitions=5)
        learnable_result = AUC(validation, exclude=evaluation).evaluate(
            GenericGraphFilter(optimal_params, preprocessor=pre, max_iters=10000).rank(G, training))
        heat_kernel_result = AUC(evaluation, used_for_training).evaluate(
            HeatKernel(7, preprocessor=pre, max_iters=10000).rank(G, training))
        self.assertLess(heat_kernel_result, learnable_result,
                        msg="Learnable parameters should be meaningful")
        self.assertGreater(heat_kernel_result,
                           AUC(evaluation).evaluate(HeatKernel(7, preprocessor=pre, max_iters=10000).rank(G, training)),
                           msg="Metrics correctly apply exclude filter to not skew results")

    def test_chebyshev(self):
        from pygrank import optimize, GenericGraphFilter, HeatKernel, to_signal, preprocessor, split, AUC, Normalize
        import random
        G, groups = test_block_model_graph(nodes=600)
        group = groups[0]
        random.seed(1)
        used_for_training, evaluation = split(to_signal(G, {v: 1 for v in group}), training_samples=0.5)
        training, validation = split(used_for_training, training_samples=0.5)

        pre = preprocessor("symmetric", True)
        with self.assertRaises(Exception):
            optimize(lambda params: -AUC(validation, exclude=training).evaluate(
                Normalize("sum",
                          GenericGraphFilter(params, coefficient_type="unknown", preprocessor=pre, max_iters=10000)).rank(G, training)),
                     max_vals=[1] * 5, min_vals=[0] * 5, deviation_tol=0.01, divide_range=2, verbose=False, partitions=5)

        optimal_params = optimize(lambda params: -AUC(validation, exclude=training).evaluate(
            Normalize("sum",
                      GenericGraphFilter(params, coefficient_type="Chebyshev", preprocessor=pre, max_iters=10000)).rank(G, training)),
                                  max_vals=[1] * 5, min_vals=[0] * 5, deviation_tol=0.01, divide_range=2, verbose=False,
                                  partitions=5)
        learnable_result = AUC(validation, exclude=evaluation).evaluate(
            GenericGraphFilter(optimal_params, preprocessor=pre, max_iters=10000).rank(G, training))
        heat_kernel_result = AUC(evaluation, exclude=used_for_training).evaluate(
            HeatKernel(7, preprocessor=pre, max_iters=10000).rank(G, training))
        self.assertLess(heat_kernel_result, learnable_result,
                        msg="Learnable parameters should be meaningful")
        self.assertGreater(heat_kernel_result,
                           AUC(evaluation).evaluate(HeatKernel(7, preprocessor=pre, max_iters=10000).rank(G, training)),
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
        pg.load_backend("tensorflow")
        self.assertEqual(pg.backend_name(), "tensorflow")
        self.test_backend_conversion()
        self.test_pagerank()
        self.test_venuerank()
        self.test_absorbing_walk()
        self.test_prevent_passing_node_lists_as_graphs()
        pg.load_backend("numpy")
        self.assertEqual(pg.backend_name(), "numpy")
        with self.assertRaises(Exception):
            pg.load_backend("unknown")
        self.assertEqual(pg.backend_name(), "numpy")
        self.test_pagerank()

    def test_rank_order_convergence(self):
        graph = test_graph()
        algorithm1 = pg.Ordinals(pg.PageRank(0.85, tol=1.E-12, max_iters=1000))
        algorithm2 = pg.Ordinals(pg.PageRank(0.85, convergence=pg.RankOrderConvergenceManager(0.85)))
        algorithm3 = pg.Ordinals(pg.PageRank(0.85, convergence=pg.RankOrderConvergenceManager(0.85, 0.99, "fraction_of_walks")))
        ranks1 = algorithm1.rank(graph, {"A": 1})
        ranks2 = algorithm2.rank(graph, {"A": 1})
        ranks3 = algorithm3.rank(graph, {"A": 1})
        self.assertGreater(pg.SpearmanCorrelation(ranks1)(ranks2), 0.95)
        self.assertGreater(pg.SpearmanCorrelation(ranks1)(ranks3), 0.95)
        self.assertGreater(pg.SpearmanCorrelation(ranks3)(ranks2), 0.95)
        self.assertTrue("17 iterations" in str(algorithm3.ranker.convergence))

        with self.assertRaises(Exception):
            algorithm = pg.Ordinals(pg.PageRank(0.85, convergence=pg.RankOrderConvergenceManager(0.85, 0.99, "unknown")))
            algorithm.rank(graph, {"A": 1})
            
    