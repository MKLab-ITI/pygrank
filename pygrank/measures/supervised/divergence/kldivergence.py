from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import backend, GraphSignalData, BackendPrimitive


class KLDivergence(Supervised):
    """Computes the KL-divergence of given vs known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores, normalization=True)
        eps = backend.epsilon()
        known_scores = known_scores + eps
        known_scores = backend.safe_div(known_scores, backend.sum(known_scores))
        scores = scores + eps
        scores = backend.safe_div(scores, backend.sum(scores))
        ratio = scores / known_scores
        ret = backend.sum(scores * backend.log(ratio))
        return ret
