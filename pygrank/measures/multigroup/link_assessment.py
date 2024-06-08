import numpy as np
import warnings
from pygrank.measures import AUC
from pygrank.measures.multigroup.clustering_coefficient import (
    _cos_similarity,
    _dot_similarity,
)


class LinkAssessment:
    """Normalizes scores by dividing with their maximal value."""

    def __init__(
        self,
        graph,
        nodes=None,
        measure=AUC,
        similarity="cos",
        hops=1,
        max_positive_samples=2000,
        max_negative_samples=2000,
        seed=0,
        progress=lambda x: x,
    ):
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
        if self.G.is_directed():  # pragma: no cover
            warnings.warn(
                "LinkAssessment is designed for undirected graphs", stacklevel=2
            )
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
            positive_candidates = np.random.choice(
                positive_candidates, self.max_positive_samples
            )
        negative_candidates = list(self.G)
        real = list()
        predicted = list()
        weights = list()
        for node in self._progress(positive_candidates):
            neighbors = {node: 0.0}
            pending = [node]
            while len(pending) != 0:
                next_node = pending.pop()
                hops = neighbors[next_node]
                Gneighbors = set(self.G.neighbors(next_node))
                if hops < self.hops:
                    for neighbor in Gneighbors:
                        if neighbor not in neighbors:
                            neighbors[neighbor] = hops + 1
                            pending.append(neighbor)
            for positive in neighbors:
                if positive != node:
                    real.append(1)
                    predicted.append(self._similarity(node, positive, scores))
                    weights.append(1)
                    # weights.append(1.-(neighbors[positive]-1)/self.hops)
            for negative in np.random.choice(
                negative_candidates,
                min(self.max_negative_samples, len(negative_candidates)),
            ):
                if negative != node and negative not in neighbors:
                    real.append(0)
                    predicted.append(self._similarity(node, negative, scores))
                    weights.append(1)
        return self.measure(real)(predicted)

    def __call__(self, scores):
        return self.evaluate(scores)
