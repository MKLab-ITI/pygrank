from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import backend, GraphSignalData, BackendPrimitive
import numpy as np


class OrderAccuracy(Supervised):
    """Computes the order of the base scores and then checks whether new ones maintain it."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        known_order = np.argsort(backend.to_numpy(known_scores))
        correct = 0
        n = len(known_order) - 1
        irrelevant = 0
        for i in range(len(known_order) - 1):
            u = known_order[i]
            v = known_order[i + 1]
            if scores[u] < scores[v]:
                correct += 1
            elif scores[u] == scores[v]:
                irrelevant += 1
        return backend.safe_div(correct, n - irrelevant)
