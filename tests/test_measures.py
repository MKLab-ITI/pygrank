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
        from pygrank import pRule, KLDivergence
        self.assertEqual(pRule([0])([0]), 0)
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
