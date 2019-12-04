import numpy as np
import sklearn.metrics
import warnings


class NDCG:
    """Provides evaluation of NDCG@k score between given and known ranks"""

    def __init__(self, known_ranks, k=None):
        """ Initializes the PageRank scheme parameters.

        Attributes:
            known_ranks: A dict of known ranks, where higher ranks correspond to more related elements.
            k: Optional. Calculates NDCG@k. If None (default), len(known_ranks) is used.
        """
        self.known_ranks = known_ranks
        if not k is None and k > len(known_ranks):
            raise Exception("NDCG@k cannot be computed for k greater than the number of known ranks")
        self.k = len(known_ranks) if k is None else k

    def evaluate(self, ranks):
        DCG = 0
        IDCG = 0
        for i, v in enumerate(list(sorted(ranks, key=ranks.get, reverse=True))[:self.k]):
            DCG += self.known_ranks.get(v, 0) / np.log2(i + 2)
        for i, v in enumerate(list(sorted(self.known_ranks, key=self.known_ranks.get, reverse=True))[:self.k]):
            IDCG += self.known_ranks[v] / np.log2(i + 2)
        return DCG / IDCG


class AUC:
    """Wrapper for sklearn.metrics.auc evaluation"""

    def __init__(self, known_ranks):
        self.known_ranks = known_ranks

    def evaluate(self, ranks):
        fpr, tpr, _ = sklearn.metrics.roc_curve([self.known_ranks.get(v, 0) for v in ranks], list(ranks.values()))
        return sklearn.metrics.auc(fpr, tpr)
