class SeedOversampling:
    def __init__(self, ranker, method='safe'):
        self.ranker = ranker
        self.method = method.lower()

    def rank(self, G, personalization):
        pygrank.algorithms.utils.assert_binary(personalization)
        if self.method == 'safe':
            ranks = self.ranker.rank(G, personalization)
            threshold = min(ranks[u] for u in personalization if personalization[u] == 1)
            personalization = {v: 1 for v in G.nodes() if ranks[v] >= threshold}
        elif self.method == 'neighbors':
            for u in [u for u in personalization if personalization[u] == 1]:
                for v in G.neighbors(u):
                    personalization[v] = 1
        else:
            raise Exception("Supported oversampling methods: safe, neighbors")
        return self.ranker.rank(G, personalization)


class BoostedSeedOversampling:
    def __init__(self, ranker, objective='partial', oversample_from_iteration='previous', weight_convergence_manager=None):
        self.ranker = ranker
        self.objective = objective.lower()
        self.oversample_from_iteration = oversample_from_iteration.lower()
        self.weight_convergence = pygrank.algorithms.utils.ConvergenceManager(error_type="small_value", tol=0.001, max_iters=100) if weight_convergence_manager is None else weight_convergence_manager

    def _boosting_weight(self, r0_N, Rr0_N, RN):
        if self.objective == 'partial':
            a_N = sum(r0_N[u]*Rr0_N[u]**2 for u in r0_N)/float(sum(Rr0_N[u]**2 for u in Rr0_N)) \
                  -sum(r0_N[u]*RN.get(u,0)*Rr0_N[u] for u in r0_N)/float(sum(Rr0_N[u]**2 for u in Rr0_N))
        elif self.objective == 'naive':
            a_N = 0.5-0.5*sum(r0_N[u]*RN.get(u,0)*Rr0_N[u] for u in r0_N)/float(sum(Rr0_N[u]**2 for u in Rr0_N))
        else:
            raise Exception("Supported boosting objectives: partial, naive")
        return a_N

    def rank(self, G, personalization):
        r0_N = personalization.copy()
        RN = self.ranker.rank(G, r0_N)
        a_N = 1
        suma_N = 1
        self.weight_convergence.start()
        while not self.weight_convergence.has_converged(a_N):
            if self.oversample_from_iteration == 'previous':
                threshold = min(RN[u] for u in r0_N if r0_N[u] == 1)
            elif self.oversample_from_iteration == 'original':
                threshold = min(RN[u] for u in personalization if personalization[u] == 1)
            else:
                raise Exception("Boosting only supports oversampling from iterations: previous, original")
            r0_N = {u: 1 for u in RN if RN[u] >= threshold}
            Rr0_N = self.ranker.rank(G, r0_N)
            a_N = self._boosting_weight(r0_N, Rr0_N, RN)
            for u in G.nodes():
                RN[u] = RN.get(u,0) + a_N*Rr0_N[u]
            suma_N += a_N
        return RN