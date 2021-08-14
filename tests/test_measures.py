import unittest
from tests.example_graph import test_graph, test_block_model_graph


class Test(unittest.TestCase):

    def test_auc_ndcg_compliance(self):
        from pygrank import PageRank, HeatKernel, AUC, NDCG
        G, groups = test_block_model_graph()
        group = groups[0]
        p = {v: 1 for v in group[:len(group)//2]}
        scores1 = PageRank()(G, p)
        scores2 = HeatKernel()(G, p)
        AUC1 = AUC({v: 1 for v in group[len(group)//2:]}, exclude=group[:len(group)//2])(scores1)
        AUC2 = AUC({v: 1 for v in group[len(group)//2:]}, exclude=group[:len(group)//2])(scores2)
        NDCG1 = NDCG({v: 1 for v in group[len(group)//2:]}, exclude=group[:len(group)//2])(scores1)
        NDCG2 = NDCG({v: 1 for v in group[len(group)//2:]}, exclude=group[:len(group)//2])(scores2)
        self.assertTrue((AUC1 < AUC2) == (NDCG1 < NDCG2))

        with self.assertRaises(Exception):
            NDCG({v: 1 for v in group[len(group)//2:]}, exclude=group[:len(group)//2], k=len(G)+1)(scores2)

    def test_edge_cases(self):
        from pygrank import pRule, KLDivergence, AUC
        self.assertEqual(pRule([0])([0]), 0)
        with self.assertRaises(Exception):
            AUC([0, 0, 0])([0, 1, 0])
        with self.assertRaises(Exception):
            AUC([1, 1, 1])([0, 1, 0])
        with self.assertRaises(Exception):
            KLDivergence([0])([-1])
        with self.assertRaises(Exception):
            KLDivergence([0], exclude={"A": 1})([1])

    def test_strange_measure_input_types(self):
        from pygrank import PageRank, NDCG, to_signal
        G, groups = test_block_model_graph()
        group = groups[0]
        p = {v: 1 for v in group[:len(group)//2]}
        scores = PageRank()(G, p)
        NDCG(to_signal(scores, {v: 1 for v in group[:len(group)//2]}), k=3)({v: scores[v] for v in scores})

    def test_accuracy(self):
        from pygrank import Accuracy, Mabs
        self.assertEqual(Accuracy([1, 2, 3])([1, 2, 3]), 1)
        self.assertEqual(Mabs([3, 1, 1])([2, 0, 2]), 1)

    def test_cross_entropy(self):
        from pygrank import CrossEntropy
        self.assertAlmostEqual(CrossEntropy([1, 1, 1])([1, 1, 1]), 0, places=12)

    def test_benchmark_print(self):
        import pygrank as pg
        self.assertEqual(pg.benchmark.utils._fraction2str(0.1), ".10")
        self.assertEqual(pg.benchmark.utils._fraction2str(0.00001), "0")
        self.assertEqual(pg.benchmark.utils._fraction2str(1), "1.00")
        pg.benchmark_print(pg.supervised_benchmark(pg.create_demo_filters(), pg.load_datasets(["ant"])))
        ret = pg.benchmark_dict(pg.supervised_benchmark(pg.create_demo_filters(), pg.load_datasets(["ant"])))
        self.assertTrue(isinstance(ret, dict))
        self.assertTrue(isinstance(ret["ant"], dict))

    def test_unsupervised_edge_cases(self):
        import pygrank as pg
        import networkx as nx
        self.assertEqual(pg.Density(nx.Graph())([]), 0)
        self.assertEqual(pg.Modularity(nx.Graph())([]), 0)

    def test_unsupervised_vs_auc(self):
        import pygrank as pg
        algorithms = pg.create_variations(pg.create_many_filters(), pg.create_many_variation_types())
        auc_scores = pg.benchmark_scores(pg.supervised_benchmark(algorithms, pg.load_datasets(["ant"]), pg.AUC))
        self.assertGreater(sum(pg.benchmark_scores(pg.supervised_benchmark(algorithms, pg.load_datasets(["ant"]), "time"))), 0)
        conductance_scores = pg.benchmark_scores(pg.supervised_benchmark(algorithms, pg.load_datasets(["ant"]),
                                                                         lambda _, __: pg.Conductance()))
        density_scores = pg.benchmark_scores(pg.supervised_benchmark(algorithms, pg.load_datasets(["ant"]),
                                                                         lambda _, __: pg.Density()))
        modularity_scores = pg.benchmark_scores(pg.supervised_benchmark(algorithms, pg.load_datasets(["ant"]),
                                                                         lambda _, __: pg.Modularity(max_positive_samples=100)))
        self.assertLess(pg.PearsonCorrelation(auc_scores)(conductance_scores), -0.6)
        self.assertLess(pg.SpearmanCorrelation(auc_scores)(density_scores), 0)
        pg.SpearmanCorrelation(auc_scores)(modularity_scores)

    def test_aggregated(self):
        import pygrank as pg
        y1 = [1, 1, 0]
        y2 = [1, 0, 0]
        y3 = [1, 1, 0]
        self.assertEqual(pg.GM().add(pg.AUC(y1), max_val=0.5).add(pg.AUC(y2), min_val=0.9).evaluate(y3), 0.45**0.5)
        self.assertEqual(pg.AM().add(pg.AUC(y1), max_val=0.5).add(pg.AUC(y2), min_val=0.9).evaluate(y3), 0.7)

