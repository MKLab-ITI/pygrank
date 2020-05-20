import numpy as np
import warnings
import sklearn.metrics
import networkx as nx
import tqdm


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


class LinkAUC:
    """ Normalizes ranks by dividing with their maximal value.

    Attributes:
        ranker: Optional. The ranking algorithm.
        nodes: The list of nodes whose edges are used in for evaluation. If None (default) all graph nodes are used.
    """
    def __init__(self, G, nodes=None, evaluation="AUC", similarity="cos", max_positive_samples=2000, max_negative_samples=2000, hops=1, seed=1):
        self.G = G
        self.nodes = list(G) if nodes is None else list(set(list(nodes)))
        self.max_positive_samples = max_positive_samples
        self.max_negative_samples = max_negative_samples
        self.hops = hops
        self.seed = seed
        self.evaluation = evaluation
        if self.G.is_directed():
            warnings.warn("LinkAUC is designed for undirected graphs", stacklevel=2)
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
        negative_candidates = list(self.G)
        if len(negative_candidates) > self.max_negative_samples:
            negative_candidates = np.random.choice(negative_candidates, self.max_negative_samples)
        real = list()
        predicted = list()
        if self.hops == -1:
            for v in positive_candidates:
                for u in self.G._adj[v]:
                    real.append(1)
                    predicted.append(self._similarity(v, u, ranks))
                    for negative in self.G._adj[v]:
                        if not self.G.has_edge(u, negative):
                            real.append(0)
                            predicted.append(self._similarity(u, negative, ranks))
                    for negative in self.G._adj[u]:
                        if not self.G.has_edge(v, negative):
                            real.append(0)
                            predicted.append(self._similarity(v, negative, ranks))
                    fpr, tpr, _ = sklearn.metrics.roc_curve(real, predicted)
        else:
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
                for negative in negative_candidates:
                    if negative != node and negative not in neighbors:
                        real.append(0)
                        predicted.append(self._similarity(node, negative, ranks))
                        weights.append(1)
        if self.evaluation == "AUC":
            fpr, tpr, _ = sklearn.metrics.roc_curve(real, predicted, sample_weight=weights)
            return sklearn.metrics.auc(fpr, tpr)
        elif self.evaluation == "CrossEntropy":
            return sum(-weights[i]*(np.log(predicted[i]+1.E-12) if real[i] == 1 else np.log(1-predicted[i]+1.E-12)) for i in range(len(real)))
        else:
            raise Exception("Invalid evaluation function (only AUC and CrossEntropy are accepted)")


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



class MultiUnsupervised:
    def __init__(self, metric_type, G, **kwargs):
        self.metric = metric_type(G, **kwargs)

    def evaluate(self, ranks):
        evaluations = [self.metric.evaluate(group_ranks) for group_ranks in ranks.values()]
        return sum(evaluations) / len(evaluations)


class MultiSupervised:
    def __init__(self, metric_type, ground_truth):
        self.metrics = {group_id: metric_type(group_truth) for group_id, group_truth in ground_truth.items()}

    def evaluate(self, ranks):
        evaluations = [self.metrics[group_id].evaluate(group_ranks) for group_id, group_ranks in ranks.items()]
        return sum(evaluations) / len(evaluations)
