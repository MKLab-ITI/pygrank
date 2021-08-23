import unittest
import pygrank as pg


class Test(unittest.TestCase):

    def test_autotune(self):
        _, G, groups = next(pg.load_datasets_multiple_communities(["bigraph"]))
        group = groups[0]
        training, evaluation = pg.split(pg.to_signal(G, {v: 1 for v in group}), training_samples=0.5)
        auc1 = pg.AUC(evaluation, exclude=training)(pg.PageRank().rank(training))
        auc2 = pg.AUC(evaluation, exclude=training)(pg.ParameterTuner(optimization_dict=dict()).rank(training))
        self.assertLessEqual(auc1, auc2, "Autotune should find good parameters")

    def test_autotune_manual(self):
        _, G, groups = next(pg.load_datasets_multiple_communities(["bigraph"]))
        group = groups[0]
        training, evaluation = pg.split(pg.to_signal(G, {v: 1 for v in group}), training_samples=0.5)
        auc1 = pg.AUC(evaluation, exclude=training)(pg.PageRank().rank(training))
        alg2 = pg.ParameterTuner(lambda params: pg.PageRank(params[0]), max_vals=[0.99], min_vals=[0.5]).tune(training)
        auc2 = pg.AUC(evaluation, exclude=training)(alg2.rank(training))
        self.assertLessEqual(auc1, auc2, "Autotune should find good parameters")

    def test_autotune_methods(self):
        _, G, groups = next(pg.load_datasets_multiple_communities(["bigraph"]))
        group = groups[0]
        training, evaluation = pg.split(pg.to_signal(G, {v: 1 for v in group}), training_samples=0.5)
        auc1 = max(pg.AUC(evaluation, exclude=training)(ranker.rank(training)) for ranker in pg.create_demo_filters().values())
        auc2 = pg.AUC(evaluation, exclude=training)(pg.AlgorithmSelection(tuning_backend="tensorflow").rank(training))
        self.assertLessEqual(auc1, auc2, "Autotune should find best method")