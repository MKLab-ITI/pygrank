import numpy as np
import warnings
from pygrank.core import backend


def _cos_similarity(v, u, scores):
    dot = 0
    l2v = 0
    l2u = 0
    for group_scores in scores.values():
        ui = group_scores.get(u, 0)
        vi = group_scores.get(v, 0)
        l2u += ui * ui
        l2v += vi * vi
        dot = ui * vi
    return backend.safe_div(dot, np.sqrt(l2u * l2v))


def _dot_similarity(v, u, scores):
    dot = 0
    for group_scores in scores.values():
        ui = group_scores.get(u, 0)
        vi = group_scores.get(v, 0)
        dot = ui * vi
    return dot


class ClusteringCoefficient:
    """https://www.albany.edu/~ravi/pdfs/opsahl_etal_2009.pdf"""

    def __init__(self, G, similarity="cos", max_positive_samples=2000, seed=1):
        self.G = G
        self.max_positive_samples = max_positive_samples
        self.seed = seed
        if self.G.is_directed():  # pragma: no cover
            warnings.warn(
                "ClusteringCoefficient is designed for undirected graphs", stacklevel=2
            )
        if similarity == "cos":
            similarity = _cos_similarity
        elif similarity == "dot":
            similarity = _dot_similarity
        self._similarity = similarity

    def evaluate(self, scores):
        np.random.seed(self.seed)
        positive_candidates = list(self.G)
        if len(positive_candidates) > self.max_positive_samples:
            positive_candidates = np.random.choice(
                positive_candidates, self.max_positive_samples
            )
        existing_triplet_values = 0.0
        total_triplet_values = 0
        for v in positive_candidates:
            for u1 in self.G.neighbors(v):
                for u2 in self.G.neighbors(v):
                    """
                    value = self._similarity(u1, u2, scores)*self._similarity(v, u2, scores)*self._similarity(v, u2, scores)
                    if u2 in self.G.neighbors(u1):
                        existing_triplet_values += value
                    total_triplet_values += value
                    """
                    if u2 in self.G.neighbors(u1):
                        total_triplet_values += 1
                    existing_triplet_values += self._similarity(u1, u2, scores)
        return (
            0
            if total_triplet_values == 0
            else existing_triplet_values / total_triplet_values
        )

    def __call__(self, scores):
        return self.evaluate(scores)
