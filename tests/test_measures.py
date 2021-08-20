import unittest
from tests.example_graph import test_graph, test_block_model_graph
import pygrank as pg
import networkx as nx

class Test(unittest.TestCase):

    def test_auc_ndcg_compliance(self):
        G, groups = test_block_model_graph()
        group = groups[0]
        p = {v: 1 for v in group[:len(group)//2]}
        scores1 = pg.PageRank()(G, p)
        scores2 = pg.HeatKernel()(G, p)
        AUC1 = pg.AUC({v: 1 for v in group[len(group)//2:]}, exclude=group[:len(group)//2])(scores1)
        AUC2 = pg.AUC({v: 1 for v in group[len(group)//2:]}, exclude=group[:len(group)//2])(scores2)
        NDCG1 = pg.NDCG({v: 1 for v in group[len(group)//2:]}, exclude=group[:len(group)//2])(scores1)
        NDCG2 = pg.NDCG({v: 1 for v in group[len(group)//2:]}, exclude=group[:len(group)//2])(scores2)
        self.assertTrue((AUC1 < AUC2) == (NDCG1 < NDCG2))

        with self.assertRaises(Exception):
            pg.NDCG({v: 1 for v in group[len(group)//2:]}, exclude=group[:len(group)//2], k=len(G)+1)(scores2)

    def test_edge_cases(self):
        self.assertEqual(pg.pRule([0])([0]), 0)
        with self.assertRaises(Exception):
            pg.Measure()([0, 1, 0])
        with self.assertRaises(Exception):
            pg.AUC([0, 0, 0])([0, 1, 0])
        with self.assertRaises(Exception):
            pg.AUC([1, 1, 1])([0, 1, 0])
        with self.assertRaises(Exception):
            pg.KLDivergence([0])([-1])
        with self.assertRaises(Exception):
            pg.KLDivergence([0], exclude={"A": 1})([1])

    def test_strange_measure_input_types(self):
        G, groups = test_block_model_graph()
        group = groups[0]
        p = {v: 1 for v in group[:len(group)//2]}
        scores = pg.PageRank()(G, p)
        pg.NDCG(pg.to_signal(scores, {v: 1 for v in group[:len(group)//2]}), k=3)({v: scores[v] for v in scores})

    def test_accuracy(self):
        self.assertEqual(pg.Accuracy([1, 2, 3])([1, 2, 3]), 1)
        self.assertEqual(pg.Mabs([3, 1, 1])([2, 0, 2]), 1)

    def test_cross_entropy(self):
        self.assertAlmostEqual(pg.CrossEntropy([1, 1, 1])([1, 1, 1]), 0, places=12)

    def test_benchmark_print(self):
        self.assertEqual(pg.benchmark.utils._fraction2str(0.1), ".10")
        self.assertEqual(pg.benchmark.utils._fraction2str(0.00001), "0")
        self.assertEqual(pg.benchmark.utils._fraction2str(1), "1.00")
        pg.benchmark_print(pg.supervised_benchmark(pg.create_demo_filters(), pg.load_datasets_one_community(["ant"])))
        ret = pg.benchmark_dict(pg.supervised_benchmark(pg.create_demo_filters(), pg.load_datasets_one_community(["ant"])))
        self.assertTrue(isinstance(ret, dict))
        self.assertTrue(isinstance(ret["ant"], dict))

    def test_unsupervised_edge_cases(self):
        self.assertEqual(pg.Density(nx.Graph())([]), 0)
        self.assertEqual(pg.Modularity(nx.Graph())([]), 0)

    def test_unsupervised_vs_auc(self):
        algorithms = pg.create_variations(pg.create_many_filters(), pg.create_many_variation_types())
        auc_scores = pg.benchmark_scores(pg.supervised_benchmark(algorithms, pg.load_datasets_one_community(["ant"]), pg.AUC))
        self.assertGreater(sum(pg.benchmark_scores(pg.supervised_benchmark(algorithms, pg.load_datasets_one_community(["ant"]), "time"))), 0)
        conductance_scores = pg.benchmark_scores(pg.supervised_benchmark(algorithms, pg.load_datasets_one_community(["ant"]),
                                                                         lambda _, __: pg.Conductance()))
        density_scores = pg.benchmark_scores(pg.supervised_benchmark(algorithms, pg.load_datasets_one_community(["ant"]),
                                                                     lambda _, __: pg.Density()))
        modularity_scores = pg.benchmark_scores(pg.supervised_benchmark(algorithms, pg.load_datasets_one_community(["ant"]),
                                                                        lambda _, __: pg.Modularity(max_positive_samples=100)))
        pg.PearsonCorrelation(auc_scores)(conductance_scores)
        pg.SpearmanCorrelation(auc_scores)(density_scores)
        pg.SpearmanCorrelation(auc_scores)(modularity_scores)

    def test_aggregated(self):
        y1 = [1, 1, 0]
        y2 = [1, 0, 0]
        y3 = [1, 1, 0]
        self.assertEqual(pg.GM().add(pg.AUC(y1), max_val=0.5).add(pg.AUC(y2), min_val=0.9).evaluate(y3), 0.45**0.5)
        self.assertEqual(pg.AM().add(pg.AUC(y1), max_val=0.5).add(pg.AUC(y2), min_val=0.9).evaluate(y3), 0.7)

    def test_split(self):
        data = {"community1": ["A", "B", "C", "D"], "community2": ["B", "E", "F", "G", "H", "I"]}
        training, test = pg.split(data, 1)
        self.assertTrue(training==test)
        training, test = pg.split(data, 0.5)
        self.assertEqual(len(training["community2"]), 3)
        self.assertEqual(len(training["community1"]), 2)
        self.assertEqual(len(test["community2"]), 3)
        training, test = pg.split(data, 2)
        self.assertEqual(len(training["community2"]), 2)
        self.assertEqual(len(test["community1"]), 2)

    def test_remove_edges(self):
        graph = test_graph(directed=True)
        self.assertTrue(graph.has_edge("A", "B"))
        self.assertTrue(graph.has_edge("C", "D"))
        pg.remove_intra_edges(graph, {"community1": ["A", "B"], "community2": ["D", "C"]})
        self.assertTrue(graph.has_edge("B", "C"))
        self.assertFalse(graph.has_edge("A", "B"))
        self.assertFalse(graph.has_edge("C", "D"))

    def test_benchmarks(self):
        datasets = pg.downloadable_small_datasets()
        pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
        algorithms = {
            "ppr0.85": pg.PageRank(alpha=0.85, preprocessor=pre, max_iters=10000, tol=1.E-9),
            "ppr0.99": pg.PageRank(alpha=0.99, preprocessor=pre, max_iters=10000, tol=1.E-9),
            "hk3": pg.HeatKernel(t=3, preprocessor=pre, max_iters=10000, tol=1.E-9),
            "hk5": pg.HeatKernel(t=5, preprocessor=pre, max_iters=10000, tol=1.E-9),
            "tuned": pg.ParameterTuner(preprocessor=pre, max_iters=10000, tol=1.E-9),
        }
        # algorithms = benchmark.create_variations(algorithms, {"": pg.Tautology, "+SO": pg.SeedOversampling})
        # loader = pg.load_datasets_one_community(datasets)
        # pg.supervised_benchmark(algorithms, loader, "time", verbose=True)

        loader = pg.load_datasets_one_community(datasets)
        pg.benchmark_print(pg.supervised_benchmark(algorithms, loader, pg.AUC, fraction_of_training=.8))

    def test_dataset_generation(self):
        self.assertEquals(len(pg.downloadable_datasets()), len(pg.datasets))

    def test_multigroup(self):
        datasets = pg.downloadable_small_datasets()
        pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
        algorithms = {
            "ppr0.85": pg.PageRank(alpha=0.85, preprocessor=pre, max_iters=10000, tol=1.E-9),
            "ppr0.99": pg.PageRank(alpha=0.99, preprocessor=pre, max_iters=10000, tol=1.E-9),
            "hk3": pg.HeatKernel(t=3, preprocessor=pre, max_iters=10000, tol=1.E-9),
            "hk5": pg.HeatKernel(t=5, preprocessor=pre, max_iters=10000, tol=1.E-9),
        }
        loader = pg.load_datasets_multiple_communities(datasets)
        pg.benchmark_print(pg.supervised_benchmark(algorithms, loader,
                                                   lambda ground_truth, exlude: pg.MultiSupervised(pg.AUC, ground_truth, exlude)))
        loader = pg.load_datasets_multiple_communities(datasets)
        pg.benchmark_print(pg.unsupervised_benchmark(algorithms, loader,
                                                   lambda graph: pg.LinkAssessment(graph, max_positive_samples=200, max_negative_samples=200)))
        loader = pg.load_datasets_multiple_communities(datasets)
        pg.benchmark_print(pg.unsupervised_benchmark(algorithms, loader,
                                                   lambda graph: pg.LinkAssessment(graph, hops=2, similarity="dot", max_positive_samples=200, max_negative_samples=200)))
        pg.benchmark_print(pg.unsupervised_benchmark(algorithms, loader,
                                                   lambda graph: pg.ClusteringCoefficient(graph)))
        pg.benchmark_print(pg.unsupervised_benchmark(algorithms, loader,
                                                   lambda graph: pg.MultiUnsupervised(pg.Conductance, graph)))
