from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import backend, GraphSignalData, BackendPrimitive


class PearsonCorrelation(Supervised):
    """Computes the Pearson correlation coefficient between given and known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)

        mean_known_scores = backend.safe_div(
            backend.sum(known_scores), backend.length(known_scores)
        )
        mean_scores = backend.safe_div(backend.sum(scores), backend.length(scores))

        diff_known_scores = known_scores - mean_known_scores
        diff_scores = scores - mean_scores
        numerator = backend.sum(diff_known_scores * diff_scores)

        denominator = backend.sum(diff_known_scores**2) * backend.sum(diff_scores**2)
        denominator = denominator**0.5

        return backend.safe_div(numerator, denominator)
