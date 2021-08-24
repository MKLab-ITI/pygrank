import numpy as np
from pygrank.core.signals import to_signal, NodeRanking
from pygrank.algorithms.utils import ConvergenceManager, assert_binary
from pygrank.algorithms.postprocess.postprocess import Postprocessor
from pygrank.measures import MaxDifference


class SeedOversampling(Postprocessor):
    """Performs seed oversampling on a base ranker to improve the quality of predicted seeds."""

    def __init__(self, ranker: NodeRanking, method: str = 'safe'):
        """ Initializes the class with a base ranker.

        Attributes:
            ranker: The base ranker instance.
            method: Optional. Can be "safe" (default) to oversample based on the ranking scores of a preliminary
                base ranker run or "neighbors" to oversample the neighbors of personalization nodes.

        Example:
            >>> from pygrank.algorithms import PageRank, SeedOversampling
            >>> graph, seed_nodes = ...
            >>> algorithm = SeedOversampling(PageRank(alpha=0.99))
            >>> ranks = algorithm.rank(graph, personalization={1 for v in seed_nodes})
        """
        super().__init__(ranker)
        self.method = method.lower()

    def rank(self, graph=None, personalization=None, **kwargs):
        personalization = to_signal(graph, personalization)
        graph = personalization.graph
        assert_binary(personalization)
        if self.method == 'safe':
            ranks = self.ranker.rank(graph, personalization, **kwargs)
            threshold = min(ranks[u] for u in personalization if personalization[u] == 1)
            personalization = {v: 1 for v in graph.nodes() if ranks[v] >= threshold}
            return self.ranker.rank(graph, personalization, **kwargs)
        elif self.method == 'top':
            ranks = self.ranker.rank(graph, personalization, **kwargs)
            top = int(graph.number_of_nodes() * graph.number_of_nodes() / graph.number_of_edges())
            threshold = np.sort(list(ranks.values()))[len(ranks) - top]  # get top ranks
            personalization = {v: 1 for v in graph.nodes() if ranks[v] >= threshold or personalization.get(v, 0) == 1}
            return self.ranker.rank(graph, personalization, **kwargs)
        elif self.method == 'neighbors':
            personalization = dict(personalization.items())
            for u in [u for u in personalization if personalization[u] == 1]:
                for v in graph.neighbors(u):
                    personalization[v] = 1
            return self.ranker.rank(graph, personalization, **kwargs)
        else:
            raise Exception("Supported oversampling methods: safe, neighbors, top")


class BoostedSeedOversampling(Postprocessor):
    """ Iteratively performs seed oversampling and combines found ranks by weighting them with a Boosting scheme."""

    def __init__(self, ranker: NodeRanking,
                 objective: str = 'partial',
                 oversample_from_iteration: str = 'previous',
                 weight_convergence: ConvergenceManager = None):
        """ Initializes the class with a base ranker and the boosting scheme's parameters.

        Attributes:
            ranker: The base ranker instance.
            objective: Optional. Can be either "partial" (default) or "naive".
            oversample_from_iteration: Optional. Can be either "previous" (default) to oversample the ranks of the
                previous iteration or "original" to always ovesample the given personalization.
            weight_convergence: Optional.  A ConvergenceManager that helps determine whether the weights placed on
                boosting iterations have converged. If None (default), initialized with
                ConvergenceManager(error_type=pyrgank.MaxDifference, tol=0.001, max_iters=100)

        Example:
            >>> from pygrank.algorithms import PageRank, BoostedSeedOversampling
            >>> graph, seed_nodes = ...
            >>> algorithm = BoostedSeedOversampling(PageRank(alpha=0.99))
            >>> ranks = algorithm.rank(graph, personalization={1 for v in seed_nodes})
        """
        super().__init__(ranker)
        self._objective = objective.lower()
        self._oversample_from_iteration = oversample_from_iteration.lower()
        self._weight_convergence = ConvergenceManager(error_type=MaxDifference, tol=0.001, max_iters=100) \
            if weight_convergence is None else weight_convergence

    def _boosting_weight(self, r0_N, Rr0_N, RN):
        if self._objective == 'partial':
            a_N = sum(r0_N[u]*Rr0_N[u]**2 for u in r0_N)/float(sum(Rr0_N[u]**2 for u in Rr0_N)) \
                  - sum(r0_N[u]*RN.get(u, 0)*Rr0_N[u] for u in r0_N)/float(sum(Rr0_N[u]**2 for u in Rr0_N))
        elif self._objective == 'naive':
            a_N = 0.5-0.5*sum(r0_N[u]*RN.get(u, 0)*Rr0_N[u] for u in r0_N)/float(sum(Rr0_N[u]**2 for u in Rr0_N))
        else:
            raise Exception("Supported boosting objectives: partial, naive")
        return a_N

    def rank(self, graph, personalization, **kwargs):
        personalization = to_signal(graph, personalization)
        r0_N = personalization.normalized(False)
        RN = self.ranker.rank(graph, r0_N, **kwargs)
        a_N = 1
        sum_a_N = 1
        self._weight_convergence.start()
        while not self._weight_convergence.has_converged(a_N):
            if self._oversample_from_iteration == 'previous':
                threshold = min(RN[u] for u in r0_N if r0_N[u] == 1)
            elif self._oversample_from_iteration == 'original':
                threshold = min(RN[u] for u in personalization if personalization[u] == 1)
            else:
                raise Exception("Boosting only supports oversampling from iterations: previous, original")
            r0_N = {u: 1 for u in RN if RN[u] >= threshold}
            Rr0_N = self.ranker.rank(graph, r0_N, **kwargs)
            a_N = self._boosting_weight(r0_N, Rr0_N, RN)
            RN = to_signal(RN, [RN.get(u, 0) + a_N*Rr0_N[u] for u in graph])
            sum_a_N += a_N
        return RN
