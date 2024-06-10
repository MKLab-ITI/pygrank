import numpy as np
import sklearn.metrics
from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import backend, GraphSignalData, BackendPrimitive


class AUC(Supervised):
    """Wrapper for sklearn.metrics.auc evaluation."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        if backend.min(known_scores) == backend.max(known_scores):
            raise Exception("Cannot evaluate AUC when all labels are the same")
        fpr, tpr, _ = sklearn.metrics.roc_curve(
            backend.to_numpy(known_scores), backend.to_numpy(scores)
        )
        return sklearn.metrics.auc(fpr, tpr)
