import numpy as np
import warnings

from pygrank.measures import AUC
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


class LinkAssessment:
    """ Normalizes scores by dividing with their maximal value.
    """
    def __init__(self, graph, nodes=None, measure=AUC, similarity="cos", hops=1, max_positive_samples=2000, max_negative_samples=2000, seed=0, progress=lambda x: x):
        """
        Args:
            graph: The graph on which to perform the evaluation.
            nodes: The list of nodes whose edges are used for evaluation. If None (default) all graph nodes are used.
            measure: The measure with which to assess prediction quality. Default is pygrank.AUC.
            similarity: "cos" (default) or "dot"
            hops: For the default measure, *hops=1* corresponds to LinkAUC and *hops=2* to HopAUC.
            max_positive_samples: A sampling strategy to reduce running time. Default is 2000.
            max_negative_samples: A sampling strategy to reduce running time. Default is 2000.
            seed: A randomization seed to ensure reproducibility (and comparability between experiments) of sampling
                strategies. If None, re-runing the same experiments may produce different results. Default is 0.
            progress: A wrapper to track progress as it iterates through a list (e.g. lambda x: tqdm.tqdm(x, desc="links") )
        """
        self.G = graph
        self.nodes = list(graph) if nodes is None else list(set(list(nodes)))
        self.max_positive_samples = max_positive_samples
        self.max_negative_samples = max_negative_samples
        self.hops = hops
        self.seed = seed
        self.measure = measure
        if self.G.is_directed():   # pragma: no cover
            warnings.warn("LinkAssessment is designed for undirected graphs", stacklevel=2)
        if similarity == "cos":
            similarity = _cos_similarity
        elif similarity == "dot":
            similarity = _dot_similarity
        self._similarity = similarity
        self._progress = progress

    def evaluate(self, scores):
        if self.seed is not None:
            np.random.seed(self.seed)
        positive_candidates = list(self.G)
        if len(positive_candidates) > self.max_positive_samples:
            positive_candidates = np.random.choice(positive_candidates, self.max_positive_samples)
        negative_candidates = list(self.G)
        real = list()
        predicted = list()
        weights = list()
        for node in self._progress(positive_candidates):
            neighbors = {node: 0.}
            pending = [node]
            while len(pending) != 0:
                next_node = pending.pop()
                hops = neighbors[next_node]
                if hops < self.hops:
                    for neighbor in self.G._adj[next_node]:
                        if neighbor not in neighbors:
                            neighbors[neighbor] = hops + 1
                            pending.append(neighbor)
            for positive in neighbors:
                if positive != node:
                    real.append(1)
                    predicted.append(self._similarity(node, positive, scores))
                    weights.append(1)
                    #weights.append(1.-(neighbors[positive]-1)/self.hops)
            for negative in np.random.choice(negative_candidates, min(self.max_negative_samples, len(negative_candidates))):
                if negative != node and negative not in neighbors:
                    real.append(0)
                    predicted.append(self._similarity(node, negative, scores))
                    weights.append(1)
        return self.measure(real)(predicted)

    def __call__(self, scores):
        return self.evaluate(scores)


class ClusteringCoefficient:
    """https://www.albany.edu/~ravi/pdfs/opsahl_etal_2009.pdf"""
    def __init__(self, G, similarity="cos", max_positive_samples=2000, seed=1):
        self.G = G
        self.max_positive_samples = max_positive_samples
        self.seed = seed
        if self.G.is_directed():   # pragma: no cover
            warnings.warn("ClusteringCoefficient is designed for undirected graphs", stacklevel=2)
        if similarity == "cos":
            similarity = _cos_similarity
        elif similarity == "dot":
            similarity = _dot_similarity
        self._similarity = similarity

    def evaluate(self, scores):
        np.random.seed(self.seed)
        positive_candidates = list(self.G)
        if len(positive_candidates) > self.max_positive_samples:
            positive_candidates = np.random.choice(positive_candidates, self.max_positive_samples)
        existing_triplet_values = 0.
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
        return 0 if total_triplet_values == 0 else existing_triplet_values / total_triplet_values

    def __call__(self, scores):
        return self.evaluate(scores)


class MultiUnsupervised:
    def __init__(self, metric_type, G, **kwargs):
        self.metric = metric_type(G, **kwargs)

    def evaluate(self, scores):
        evaluations = [self.metric.evaluate(group_scores) for group_scores in scores.values()]
        return sum(evaluations) / len(evaluations)

    def __call__(self, scores):
        return self.evaluate(scores)


class MultiSupervised:
    def __init__(self, metric_type, ground_truth, exclude=None):
        self.metrics = {group_id: metric_type(group_truth, exclude[group_id] if exclude is not None else None) for group_id, group_truth in ground_truth.items()}

    def evaluate(self, scores):
        evaluations = [self.metrics[group_id].evaluate(group_scores) for group_id, group_scores in scores.items()]
        return sum(evaluations) / len(evaluations)
    
    def __call__(self, scores):
        return self.evaluate(scores)
