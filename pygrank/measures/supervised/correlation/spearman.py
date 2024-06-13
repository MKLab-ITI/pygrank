from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import backend, GraphSignalData, BackendPrimitive
import scipy.stats


class SpearmanCorrelation(Supervised):
    """Computes the Spearman correlation coefficient between given and known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        if len(known_scores) == 0:
            return 1
        return scipy.stats.spearmanr(
            backend.to_numpy(known_scores), backend.to_numpy(scores)
        )[0]
