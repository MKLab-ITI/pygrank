from pygrank.algorithms.autotune import optimize
from pygrank.algorithms.postprocess.postprocess import Tautology, Normalize, Postprocessor
from pygrank.core.signals import GraphSignal, to_signal
from pygrank.measures import pRule
from pygrank import backend


class FairPersonalizer(Postprocessor):
    """
    A personalization editing scheme that aims to edit graph signal priors (i.e. personalization) to produce
    disparate
    """

    def __init__(self, ranker, target_pRule=1, retain_rank_weight=1, pRule_weight=1, error_type="KL",
                 parameter_buckets=1, max_residual=1):
        """
        Instantiates a personalization editing scheme that trains towards optimizing
        retain_rank_weight*error_type(original scores, editing-induced scores)
            + pRule_weight*min(induced score pRule, target_pRule)

        Args:
            ranker: The base ranking algorithm.
            target_pRule: Up to which value should pRule be improved. pRule values greater than this are not penalized
                further.
            retain_rank_weight: Can be used to penalize deviations from original posteriors due to editing.
                Use the default value 1 unless there is a specific reason to scale the error. Higher values
                correspond to tighter maintenance of original posteriors, but may not improve fairness as much.
            pRule_weight: Can be used to penalize low pRule values. Either use the default value 1 or, if you want to
                place most emphasis on pRule maximization (instead of trading-off between fairness and posterior
                preservation) 10 is a good empirical starting point.
            error_type: The error type used to penalize deviations from original posterior scores. "KL" (default) uses
                KL-divergence and is used in [krasanakis2020prioredit]. "mabs" uses the mean absolute error and is used
                in the earlier [krasanakis2020fairconstr]. The latter does not maintain fairness as well on average,
                but is sometimes better for specific graphs.
            parameter_buckets: How many sets of parameters to be used to . Default is 1. More parameters could be needed to
                to track, but running time scales **exponentially** to these (with base 4).
            max_residual: An upper limit on how much the original personalization is preserved, i.e. a fraction of
                it in the range [0, max_residual] is preserved. Default is 1 and is introduced by [krasanakis2020prioredit],
                but 0 can be used for exact replication of [krasanakis2020fairconstr].
        """
        super().__init__(ranker)
        self.target_pRule = target_pRule
        self.retain_rank_weight = retain_rank_weight
        self.pRule_weight = pRule_weight
        self.error_type = error_type
        self.parameter_buckets = parameter_buckets
        self.max_residual = max_residual

    def __culep(self, personalization, sensitive, ranks, params):
        res = personalization*params[-1]
        ranks = ranks.np / backend.max(ranks.np)
        personalization = personalization / backend.max(personalization)
        for i in range(self.parameter_buckets):
            a = sensitive*(params[0+4*i]-params[1+4*i]) + params[1+4*i]
            b = sensitive*(params[2+4*i]-params[3+4*i]) + params[3+4*i]
            if self.error_type == "mabs":
                res += (1-a)*backend.exp(b*(ranks-personalization)) + a*backend.exp(-b*(ranks-personalization))
            else:
                res += (1-a)*backend.exp(b*backend.abs(ranks-personalization)) + a*backend.exp(-b*backend.abs(ranks-personalization))
        return res

    def __prule_loss(self, ranks: GraphSignal, original_ranks: GraphSignal, sensitive: object, personalization: object) -> object:
        prule = self.pRule(ranks)
        if self.error_type == "mabs":
            ranks = ranks.np / backend.max(ranks.np)
            original_ranks = original_ranks.np / backend.max(original_ranks.np)
            return self.retain_rank_weight * backend.sum(backend.abs(ranks - original_ranks)) / backend.length(ranks) \
                  - self.pRule_weight * min(self.target_pRule, prule)
        if self.error_type == "KL":
            ranks = ranks.np / backend.sum(ranks.np)
            original_ranks = original_ranks.np / backend.sum(original_ranks.np)
            return self.retain_rank_weight * backend.dot(ranks[original_ranks != 0], -backend.log(original_ranks[original_ranks!=0]/ranks[original_ranks!=0])) - self.pRule_weight * min(self.target_pRule, prule)
        raise Exception("Invalid error type")

    def rank(self, G, personalization, sensitive, *args, **kwargs):
        personalization = to_signal(G, personalization)
        G = personalization.graph
        self.pRule = pRule(sensitive)
        sensitive, personalization = self.pRule.to_numpy(personalization)
        ranks = self.ranker.rank(G, personalization, *args, as_dict=False, **kwargs)

        def loss(params):
            fair_pers = self.__culep(personalization, sensitive, ranks, params)
            fair_ranks = self.ranker.rank(G, personalization=fair_pers, *args, as_dict=False, **kwargs)
            return self.__prule_loss(fair_ranks, ranks, sensitive, personalization)

        optimal_params = optimize(loss,
                                  max_vals=[1, 1, 10, 10] * self.parameter_buckets + [self.max_residual],
                                  min_vals=[0, 0, -10, -10]*self.parameter_buckets+[0],
                                  deviation_tol=1.E-2,
                                  divide_range=2,
                                  partitions=5)
        optimal_personalization = personalization=self.__culep(personalization, sensitive, ranks, optimal_params)
        del self.pRule
        return self.ranker.rank(G, optimal_personalization, *args, **kwargs)


class AdHocFairness(Postprocessor):
    """Adjusts node scores so that the sum of sensitive nodes is moved closer to the sum of non-sensitive ones based on
    ad hoc literature assumptions about how unfairness is propagated in graphs.
    """

    def __init__(self, ranker=None, method="O", eps=1.E-12):
        """
        Initializes the fairness-aware postprocessor.

        Args:
            ranker: The base ranking algorithm.
            method: The method with which to adjust weights. If "O" (default) an optimal gradual adjustment is performed
                [tsioutsiouliklis2020fairness].
                If "B" node scores are weighted according to whether the nodes are sensitive, so that
                the sum of sensitive node scores becomes equal to the sum of non-sensitive node scores
                [tsioutsiouliklis2020fairness].  If "fairwalk" the graph is pre-processed so that, when possible,
                walks are equally probable to visit sensitive or non-sensitive nodes at non-restarting iterations
                [rahman2019fairwalk].
            eps: A small value to consider rank redistribution to have converged. Default is 1.E-12.
        """
        if ranker is not None and not callable(getattr(ranker, "rank", None)):
            ranker, method = method, ranker
            if not callable(getattr(ranker, "rank", None)):
                ranker = None
        super.__init__(Tautology() if ranker is None else ranker)
        self.method = method
        self.eps = eps

    def __distribute(self, DR, ranks, sensitive):
        while True:
            ranks = {v: ranks[v] * sensitive.get(v, 0) for v in ranks if ranks[v] * sensitive.get(v, 0) != 0}
            d = DR / len(ranks)
            min_rank = min(ranks.values())
            if min_rank < self.eps:
                break
            if min_rank > d:
                ranks = {v: val - d for v, val in ranks.items()}
                break
            ranks = {v: val - min_rank for v, val in ranks.items()}
            DR -= len(ranks) * min_rank
        return ranks

    def __reweigh(self, graph, sensitive):
        if not getattr(self, "reweighs", None):
            self.reweighs = dict()
        if graph not in self.reweighs:
            phi = sum(sensitive.values())/len(graph)
            new_graph = graph.copy()
            for u, v, d in new_graph.edges(data=True):
                d["weight"] = 1./(sensitive[u]*phi+(1-sensitive[u])*(1-phi))
            self.reweighs[graph] = new_graph
        return self.reweighs[graph]

    def _transform(self, ranks, sensitive):
        phi = sum(sensitive.values())/len(ranks)
        if self.method == "O":
            ranks = Normalize("sum").transform(ranks)
            sumR = sum(ranks[v] * sensitive.get(v, 0) for v in ranks)
            sumB = sum(ranks[v] * (1 - sensitive.get(v, 0)) for v in ranks)
            numR = sum(sensitive.values())
            numB = len(ranks) - numR
            if sumR < phi:
                red = self.__distribute(phi - sumR, ranks, {v: 1 - sensitive.get(v, 0) for v in ranks})
                ranks = {v: red.get(v, ranks[v] + (phi - sumR) / numR) for v in ranks}
            elif sumB < 1-phi:
                red = self.__distribute(1-phi - sumB, ranks, {v: sensitive.get(v, 0) for v in ranks})
                ranks = {v: red.get(v, ranks[v] + (1-phi - sumB) / numB) for v in ranks}
        elif self.method == "B":
            sumR = sum(ranks[v]*sensitive.get(v, 0) for v in ranks)
            sumB = sum(ranks[v]*(1-sensitive.get(v, 0)) for v in ranks)
            sum_total = sumR + sumB
            sumR /= sum_total
            sumB /= sum_total
            ranks = {v: ranks[v]*(phi*sensitive.get(v, 0)/sumR+(1-phi)*(1-sensitive.get(v, 0))/sumB) for v in ranks}
        elif self.method != 'fairwalk':
            raise Exception("Invalid fairness postprocessing method", self.method)
        return ranks

    def transform(self, *args, **kwargs):
        if self.method == "fairwalk":
            raise Exception("reweighing can only occur by preprocessing the graph")
        return super().transform(*args, **kwargs)
