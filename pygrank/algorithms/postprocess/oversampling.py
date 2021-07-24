import pygrank.algorithms.utils
import numpy as np


class SeedOversampling:
    """Performs seed oversampling on a base ranker to improve the quality of predicted seeds."""

    def __init__(self, ranker, method='safe'):
        """ Initializes the class with a base ranker.

        Attributes:
            ranker: The base ranker instance.
            method: Optional. Can be "safe" (default) to oversample based on the ranking scores of a preliminary
                base ranker run or "neighbors" to oversample the neighbors of personalization nodes.

        Example:
            >>> from pygrank.algorithms.postprocess import oversampling            >>> from pygrank.algorithms import fixed
            >>> G, seed_nodes = ...
            >>> algorithm = oversampling.SeedOversampling(fixed.PageRank(alpha=0.99))
            >>> ranks = algorithm.rank(G, personalization={1 for v in seed_nodes})
        """
        self.ranker = ranker
        self.method = method.lower()

    def rank(self, G, personalization, **kwargs):
        pygrank.algorithms.utils.assert_binary(personalization)
        if self.method == 'safe':
            #prev_to_scipy = self.ranker.to_scipy
            #self.ranker.to_scipy = pygrank.algorithms.utils.MethodHasher(self.ranker.to_scipy)
            ranks = self.ranker.rank(G, personalization, **kwargs)
            threshold = min(ranks[u] for u in personalization if personalization[u] == 1)
            personalization = {v: 1 for v in G.nodes() if ranks[v] >= threshold}
            ranks = self.ranker.rank(G, personalization, **kwargs)
            #self.ranker.to_scipy = prev_to_scipy
            return ranks
        elif self.method == 'top':
            #prev_to_scipy = self.ranker.to_scipy
            #self.ranker.to_scipy = pygrank.algorithms.utils.MethodHasher(self.ranker.to_scipy)
            ranks = self.ranker.rank(G, personalization, **kwargs)
            threshold = np.sort(list(ranks.values()))[len(ranks)-int(G.number_of_nodes()*G.number_of_nodes()/G.number_of_edges())] # get top rank
            personalization = {v: 1 for v in G.nodes() if ranks[v] >= threshold or personalization.get(v,0)==1} # add only this top rank
            ranks = self.ranker.rank(G, personalization, **kwargs)
            #self.ranker.to_scipy = prev_to_scipy
            return ranks
        elif self.method == 'neighbors':
            for u in [u for u in personalization if personalization[u] == 1]:
                for v in G.neighbors(u):
                    personalization[v] = 1
            return self.ranker.rank(G, personalization, **kwargs)
        else:
            raise Exception("Supported oversampling methods: safe, neighbors, top")


class BoostedSeedOversampling:
    """ Iteratively performs seed oversampling and combines found ranks by weighting them with a Boosting scheme."""

    def __init__(self, ranker, objective='partial', oversample_from_iteration='previous', weight_convergence=None):
        """ Initializes the class with a base ranker and the boosting scheme's parameters.

        Attributes:
            ranker: The base ranker instance.
            objective: Optional. Can be either "partial" (default) or "naive".
            oversample_from_iteration: Optional. Can be either "previous" (default) to oversample the ranks of the
                previous iteration or "original" to always ovesample the given personalization.
            weight_convergence: Optional.  A ConvergenceManager that helps determine whether the weights placed on
                boosting iterations have converged. If None (default), initialized with
                ConvergenceManager(error_type="small_value", tol=0.001, max_iters=100)

        Example:
            >>> from pygrank.algorithms import fixed
            >>> from pygrank.algorithms import oversampling
            >>> G, seed_nodes = ...
            >>> algorithm = oversampling.BoostedSeedOversampling(fixed.PageRank(alpha=0.99))
            >>> ranks = algorithm.rank(G, personalization={1 for v in seed_nodes})
        """
        self.ranker = ranker
        self._objective = objective.lower()
        self._oversample_from_iteration = oversample_from_iteration.lower()
        self._weight_convergence = pygrank.algorithms.utils.ConvergenceManager(error_type="small_value", tol=0.001, max_iters=100) if weight_convergence is None else weight_convergence

    def _boosting_weight(self, r0_N, Rr0_N, RN):
        if self._objective == 'partial':
            a_N = sum(r0_N[u]*Rr0_N[u]**2 for u in r0_N)/float(sum(Rr0_N[u]**2 for u in Rr0_N)) \
                  -sum(r0_N[u]*RN.get(u,0)*Rr0_N[u] for u in r0_N)/float(sum(Rr0_N[u]**2 for u in Rr0_N))
        elif self._objective == 'naive':
            a_N = 0.5-0.5*sum(r0_N[u]*RN.get(u,0)*Rr0_N[u] for u in r0_N)/float(sum(Rr0_N[u]**2 for u in Rr0_N))
        else:
            raise Exception("Supported boosting objectives: partial, naive")
        return a_N

    def rank(self, G, personalization, **kwargs):
        #prev_to_scipy = self.ranker.to_scipy
        #self.ranker.to_scipy = pygrank.algorithms.utils.MethodHasher(self.ranker.to_scipy)
        r0_N = personalization.copy()
        RN = self.ranker.rank(G, r0_N, **kwargs)
        a_N = 1
        suma_N = 1
        self._weight_convergence.start()
        while not self._weight_convergence.has_converged(a_N):
            if self._oversample_from_iteration == 'previous':
                threshold = min(RN[u] for u in r0_N if r0_N[u] == 1)
            elif self._oversample_from_iteration == 'original':
                threshold = min(RN[u] for u in personalization if personalization[u] == 1)
            else:
                raise Exception("Boosting only supports oversampling from iterations: previous, original")
            r0_N = {u: 1 for u in RN if RN[u] >= threshold}
            Rr0_N = self.ranker.rank(G, r0_N, **kwargs)
            a_N = self._boosting_weight(r0_N, Rr0_N, RN)
            for u in G.nodes():
                RN[u] = RN.get(u,0) + a_N*Rr0_N[u]
            suma_N += a_N
        #self.ranker.to_scipy = prev_to_scipy
        return RN
