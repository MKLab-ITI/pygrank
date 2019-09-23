import numpy as np
import sklearn.metrics
import warnings


class NDCG:
    def __init__(self, known_ranks):
        self.known_ranks = known_ranks
        warnings.warn("NDCG is still under development (its implementation may be incorrect)", stacklevel=2)

    def evaluate(self, ranks):
        DCG = 0
        IDCG = 0
        for i, v in enumerate(sorted(ranks, key=ranks.get, reverse=True)):
            DCG += self.known_ranks.get(v, 0) / np.log2(i + 2)
        for i, v in enumerate(sorted(self.known_ranks, key=self.known_ranks.get, reverse=True)):
            IDCG += self.known_ranks[v] / np.log2(i + 2)
        return DCG / IDCG


class AUC:
    def __init__(self, known_ranks):
        self.known_ranks = known_ranks

    def evaluate(self, ranks):
        fpr, tpr, _ = sklearn.metrics.roc_curve([self.known_ranks.get(v, 0) for v in ranks], list(ranks.values()))
        return sklearn.metrics.auc(fpr, tpr)
