from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import backend, GraphSignalData, BackendPrimitive
import scipy.stats


class PearsonCorrelation(Supervised):
    """Computes the Pearson correlation coefficient between given and known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        return scipy.stats.pearsonr(backend.to_numpy(known_scores), backend.to_numpy(scores))[0]
