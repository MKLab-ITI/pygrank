import unittest
from tests.example_graph import test_graph, test_block_model_graph


class Test(unittest.TestCase):

    def test_auc_ndcg_compliance(self):
        from pygrank.algorithms import PageRank, HeatKernel
        from pygrank.measures import AUC, NDCG
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

