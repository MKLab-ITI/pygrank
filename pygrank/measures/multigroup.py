import numpy as np
import warnings
import sklearn.metrics
from pygrank.measures import AUC


def _cos_similarity(v, u, ranks):
    dot = 0
    l2v = 0
    l2u = 0
    for group_ranks in ranks.values():
        ui = group_ranks.get(u, 0)
        vi = group_ranks.get(v, 0)
        l2u += ui * ui
        l2v += vi * vi
        dot = ui * vi
    if l2u == 0 or l2v == 0:
        return 0
    return dot / np.sqrt(l2u * l2v)


def _dot_similarity(v, u, ranks):
    dot = 0
    for group_ranks in ranks.values():
        ui = group_ranks.get(u, 0)
        vi = group_ranks.get(v, 0)
        dot = ui * vi
    return dot


class LinkAssessment:
    """ Normalizes ranks by dividing with their maximal value.
    """
    def __init__(self, graph, nodes=None, measure=AUC, similarity="cos", hops=1, max_positive_samples=2000, max_negative_samples=2000, seed=0):
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
        """
        self.G = graph
        self.nodes = list(graph) if nodes is None else list(set(list(nodes)))
        self.max_positive_samples = max_positive_samples
        self.max_negative_samples = max_negative_samples
        self.hops = hops
        self.seed = seed
        self.measure = measure
        if self.G.is_directed():
            warnings.warn("LinkAUC is designed for undirected graphs", stacklevel=2)
        if similarity == "cos":
            similarity = _cos_similarity
        elif similarity == "dot":
            similarity = _dot_similarity
        self._similarity = similarity

    def evaluate(self, ranks):
        if self.seed is not None:
            np.random.seed(self.seed)
        positive_candidates = list(self.G)
        if len(positive_candidates) > self.max_positive_samples:
            positive_candidates = np.random.choice(positive_candidates, self.max_positive_samples)
        negative_candidates = list(self.G)
        real = list()
        predicted = list()
        weights = list()
        for node in positive_candidates:#tqdm.tqdm(positive_candidates, desc="LinkAUC"):
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
                    predicted.append(self._similarity(node, positive, ranks))
                    weights.append(1)
                    #weights.append(1.-(neighbors[positive]-1)/self.hops)
            for negative in np.random.choice(negative_candidates, min(self.max_negative_samples, len(negative_candidates))):
                if negative != node and negative not in neighbors:
                    real.append(0)
                    predicted.append(self._similarity(node, negative, ranks))
                    weights.append(1)
        return self.measure(real)(predicted)

    def __call__(self, ranks):
        return self.evaluate(ranks)


class ClusteringCoefficient:
    """https://www.albany.edu/~ravi/pdfs/opsahl_etal_2009.pdf"""
    def __init__(self, G, similarity="cos", max_positive_samples=2000, seed=1):
        self.G = G
        self.max_positive_samples = max_positive_samples
        self.seed = seed
        if self.G.is_directed():
            warnings.warn("ClusteringCoefficient is designed for undirected graphs", stacklevel=2)
        if similarity == "cos":
            self._similarity = _cos_similarity
        elif similarity == "dot":
            self._similarity = _dot_similarity
        else:
            self._similarity = similarity

    def evaluate(self, ranks):
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
                    value = self._similarity(u1, u2, ranks)*self._similarity(v, u2, ranks)*self._similarity(v, u2, ranks)
                    if u2 in self.G.neighbors(u1):
                        existing_triplet_values += value
                    total_triplet_values += value
                    """
                    if u2 in self.G.neighbors(u1):
                        total_triplet_values += 1
                    existing_triplet_values += self._similarity(u1, u2, ranks)
        if total_triplet_values == 0:
            return 0
        return existing_triplet_values / total_triplet_values

    def __call__(self, ranks):
        return self.evaluate(ranks)


class MultiUnsupervised:
    def __init__(self, metric_type, G, **kwargs):
        self.metric = metric_type(G, **kwargs)

    def evaluate(self, ranks):
        evaluations = [self.metric.evaluate(group_ranks) for group_ranks in ranks.values()]
        return sum(evaluations) / len(evaluations)

    def __call__(self, ranks):
        return self.evaluate(ranks)


class MultiSupervised:
    def __init__(self, metric_type, ground_truth, exclude=None):
        self.metrics = {group_id: metric_type(group_truth, exclude[group_id] if exclude is not None else None) for group_id, group_truth in ground_truth.items()}

    def evaluate(self, ranks):
        evaluations = [self.metrics[group_id].evaluate(group_ranks) for group_id, group_ranks in ranks.items()]
        return sum(evaluations) / len(evaluations)
    
    def __call__(self, ranks):
        return self.evaluate(ranks)
