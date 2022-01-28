import pygrank
from pygrank.algorithms.autotune import optimize
from pygrank.algorithms.postprocess.postprocess import Tautology, Normalize, Postprocessor
from pygrank.measures import pRule, Mabs, Supervised, AM, KLDivergence, TPR, TNR, Parity
from pygrank.core import GraphSignal, to_signal, backend, BackendPrimitive, NodeRanking, GraphSignalGraph, GraphSignalData
from typing import List, Optional, Union


class FairPersonalizer(Postprocessor):
    """
    A personalization editing scheme that aims to edit graph signal priors (i.e. personalization) to produce
    fairness-aware posteriors satisfying disparate impact constraints in terms of pRule.
    """

    def __init__(self,
                 ranker: NodeRanking,
                 target_pRule: float = 1,
                 retain_rank_weight: float = 1,
                 pRule_weight: float = 1,
                 error_type: Supervised = Mabs,
                 parameter_buckets: int = 1,
                 max_residual: float = 1,
                 error_skewing: bool = False,
                 parity_type: str = "impact"):
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
            error_type: The supervised measure used to penalize deviations from original posterior scores.
                pygrank.KLDivergence (default) uses is used in [krasanakis2020prioredit]. pygrank.Error is used by
                the earlier [krasanakis2020fairconstr]. The latter does not induce fairness as well on average,
                but is sometimes better for specific graphs.
            parameter_buckets: How many sets of parameters to be used to . Default is 1. More parameters could be needed to
                to track, but running time scales **exponentially** to these (with base 4).
            max_residual: An upper limit on how much the original personalization is preserved, i.e. a fraction of
                it in the range [0, max_residual] is preserved. Default is 1 and is introduced by [krasanakis2020prioredit],
                but 0 can be used for exact replication of [krasanakis2020fairconstr].
            parity_type: The type of fairness measure to be optimized. If "impact" (default) the pRule is optimized,
               if "TPR" or "TNR" the TPR and TNR parity between sensitive and non-sensitive nodes is optimized
               respectively, if "mistreatment" the AM of TPR and TNR parity is optimized.

        Example:
            >>> import pygrank as pg
            >>> graph, personalization, sensitive, algorithm = ... # sensitive is a second graph signal
            >>> algorithm = pg.FairPersonalizer(algorithm, .8, pRule_weight=10) # tries to force (weight 10) pRule to be at least 80%
            >>> ranks = algorithm.rank(graph, personalization, sensitive=sensitive)

        Example (treats class imbalanace):
            >>> import pygrank as pg
            >>> graph, personalization, algorithm = ...
            >>> algorithm = pg.FairPersonalizer(algorithm, .8, pRule_weight=10)
            >>> ranks = algorithm.rank(graph, personalization, sensitive=personalization)
        """
        super().__init__(ranker)
        self.target_pRule = target_pRule
        self.retain_rank_weight = retain_rank_weight
        self.pRule_weight = pRule_weight
        self.error_type = error_type
        self.parameter_buckets = parameter_buckets
        self.max_residual = max_residual
        self.error_skewing = error_skewing
        self.parity_type = parity_type

    def __culep(self,
                personalization: BackendPrimitive,
                sensitive: BackendPrimitive,
                ranks: BackendPrimitive,
                params: List[float]):
        ranks = ranks.np / backend.max(ranks.np)
        personalization = personalization / backend.max(personalization)
        res = ranks if self.parameter_buckets == 0 else 0
        for i in range(self.parameter_buckets):
            a = sensitive*(params[0+4*i]-params[1+4*i]) + params[1+4*i]
            b = sensitive*(params[2+4*i]-params[3+4*i]) + params[3+4*i]
            if self.error_skewing:
                res = res + (1-a)*backend.exp(b*(ranks-personalization)) + a*backend.exp(-b*(ranks-personalization))
            else:
                res = res + (1-a)*backend.exp(b*backend.abs(ranks-personalization)) + a*backend.exp(-b*backend.abs(ranks-personalization))
        return (1.0-params[-1])*res + personalization*params[-1]

    def __prule_loss(self,
                     ranks: GraphSignal,
                     original_ranks: GraphSignal,
                     sensitive: GraphSignal,
                     personalization: GraphSignal) -> object:
        prule = self.pRule(ranks)
        #ranks = ranks.np-backend.min(ranks.np)
        ranks = ranks.np / backend.max(ranks.np)
        original_ranks = original_ranks.np / backend.max(original_ranks.np)
        error = self.error_type(original_ranks)
        error_value = error(ranks)
        return -self.retain_rank_weight * error_value * error.best_direction() - self.pRule_weight * min(self.target_pRule, prule) - 0.1*prule

    def rank(self,
             G: GraphSignalGraph,
             personalization: GraphSignalData,
             sensitive: GraphSignalData, *args, **kwargs):
        personalization = to_signal(G, personalization)
        G = personalization.graph
        #self.pRule = pRule(sensitive)
        if self.parity_type == "impact":
            self.pRule = pRule(sensitive)
        elif self.parity_type == "TPR":
            self.pRule = Parity([TPR(personalization, exclude=1-sensitive.np),
                                 TPR(personalization, exclude=1-(1-sensitive.np))])
        elif self.parity_type == "TNR":
            self.pRule = Parity([TNR(personalization, exclude=1 - sensitive.np),
                                 TNR(personalization, exclude=1 - (1 - sensitive.np))])
        elif self.parity_type == "mistreatment":
            self.pRule = AM([Parity([TPR(personalization, exclude=1-sensitive.np),
                                     TPR(personalization, exclude=1-(1-sensitive.np))]),
                             Parity([TNR(personalization, exclude=1 - sensitive.np),
                                     TNR(personalization, exclude=1 - (1 - sensitive.np))])
                            ])
        else:
            raise Exception("Invalid parity type "+self.parity_type+": expected impact, TPR, TNR or mistreatment")
        sensitive, personalization = pRule(sensitive).to_numpy(personalization)
        ranks = self.ranker.rank(G, personalization, *args, **kwargs)

        def loss(params):
            fair_pers = self.__culep(personalization, sensitive, ranks, params)
            fair_ranks = self.ranker.rank(G, personalization=fair_pers, *args, **kwargs)
            return self.__prule_loss(fair_ranks, ranks, sensitive, personalization)

        optimal_params = optimize(loss,
                                  max_vals=[1, 1, 10, 10] * self.parameter_buckets + [self.max_residual],
                                  min_vals=[0, 0, -10, -10]*self.parameter_buckets+[0],
                                  deviation_tol=1.E-3,
                                  divide_range=1.5,
                                  partitions=5,
                                  depth=2)
        optimal_personalization = personalization=self.__culep(personalization, sensitive, ranks, optimal_params)
        return self.ranker.rank(G, optimal_personalization, *args, **kwargs)

    def _reference(self):
        return "fair prior editing \\cite{krasanakis2020prioredit}"


class FairTradeoff(Postprocessor):
    """
    A personalization editing scheme that aims to edit graph signal priors (i.e. personalization) to produce
    disparate
    """

    def __init__(self,
                 ranker: NodeRanking,
                 target_pRule: float = 1,
                 retain_rank_weight: float = 1,
                 pRule_weight: float = 1,
                 error_type: Supervised = Mabs):
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
            error_type: The supervised measure used to penalize deviations from original posterior scores.
                pygrank.KLDivergence (default) uses is used in [krasanakis2020prioredit]. pygrank.Error is used by
                the earlier [krasanakis2020fairconstr]. The latter does not induce fairness as well on average,
                but is sometimes better for specific graphs.
        """
        super().__init__(ranker)
        self.target_pRule = target_pRule
        self.retain_rank_weight = retain_rank_weight
        self.pRule_weight = pRule_weight
        self.error_type = error_type

    def __prule_loss(self,
                     ranks: GraphSignal,
                     original_ranks: GraphSignal) -> object:
        prule = self.pRule(ranks)
        ranks = ranks.np / backend.max(ranks.np)
        original_ranks = original_ranks.np / backend.max(original_ranks.np)
        error = self.error_type(original_ranks)
        return -self.retain_rank_weight * error(ranks) * error.best_direction() - self.pRule_weight * min(self.target_pRule, prule)

    def rank(self,
             G: GraphSignalGraph,
             personalization: GraphSignalData,
             sensitive: GraphSignalData, *args, **kwargs):
        personalization = to_signal(G, personalization)
        G = personalization.graph
        self.pRule = pRule(sensitive)
        sensitive, personalization = self.pRule.to_numpy(personalization)
        ranks = self.ranker.rank(G, personalization, sensitive=sensitive, *args, **kwargs)
        sensitive_ranks = self.ranker.rank(G, sensitive, sensitive=sensitive, *args, **kwargs)
        invert_ranks = self.ranker.rank(G, backend.max(ranks.np)-ranks.np, sensitive=sensitive,  *args, **kwargs)
        non_sensitive_ranks = self.ranker.rank(G, 1-sensitive, sensitive=sensitive,  *args, **kwargs)

        def loss(params):
            if params[0] == 0:
                return float('inf')
            fair_pers = personalization + params[0]*ranks.np  + params[1]*invert_ranks.np + params[2]*sensitive_ranks.np + params[3]*non_sensitive_ranks.np
            fair_ranks = self.ranker.rank(G, personalization=fair_pers, *args, **kwargs)
            return self.__prule_loss(to_signal(G, fair_ranks), ranks)

        optimal_params = optimize(loss,
                                  max_vals=[1, 1, 1, 1, 1],
                                  min_vals=[0, 0, 0, 0, 0],
                                  deviation_tol=1.E-2,
                                  divide_range=2,
                                  partitions=5,
                                  depth=2)
        optimal_personalization = personalization + optimal_params[0]*ranks.np + optimal_params[0]*invert_ranks.np + optimal_params[2]*sensitive_ranks.np + optimal_params[3]*non_sensitive_ranks.np
        return self.ranker.rank(G, optimal_personalization, *args, **kwargs)

    def _reference(self):
        return "fair prior editing \\cite{krasanakis2020prioredit}"


class AdHocFairness(Postprocessor):
    """Adjusts node scores so that the sum of sensitive nodes is moved closer to the sum of non-sensitive ones based on
    ad hoc literature assumptions about how unfairness is propagated in graphs.
    """

    def __init__(self,
                 ranker: Optional[Union[NodeRanking,str]] = None,
                 method: Optional[Union[NodeRanking,str]] = "O",
                 eps: float = 1.E-12):
        """
        Initializes the fairness-aware postprocessor.

        Args:
            ranker: The base ranking algorithm.
            method: The method with which to adjust weights. If "O" (default) an optimal gradual adjustment is performed
                [tsioutsiouliklis2020fairness].
                If "B" node scores are weighted according to whether the nodes are sensitive, so that
                the sum of sensitive node scores becomes equal to the sum of non-sensitive node scores
                [tsioutsiouliklis2020fairness].
            eps: A small value to consider rank redistribution to have converged. Default is 1.E-12.
        """
        if ranker is not None and not callable(getattr(ranker, "rank", None)):
            ranker, method = method, ranker
            if not callable(getattr(ranker, "rank", None)):
                ranker = None
        super().__init__(Tautology() if ranker is None else ranker)
        self.method = method
        self.eps = eps  # TODO: investigate whether using backend.epsilon() is a preferable alternative

    def __distribute(self, DR, ranks, sensitive):
        min_rank = float('inf')
        while min_rank >= self.eps and DR > 0:
            ranks = {v: ranks[v] * sensitive.get(v, 0) for v in ranks if ranks[v] * sensitive.get(v, 0) != 0}
            d = DR / len(ranks)
            min_rank = min(ranks.values())
            if min_rank > d:
                min_rank = d
            ranks = {v: val - min_rank for v, val in ranks.items()}
            DR -= len(ranks) * min_rank
        return ranks

    def _transform(self, ranks: GraphSignal, sensitive: GraphSignal):
        sensitive = to_signal(ranks, sensitive)
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
        else:
            raise Exception("Invalid fairness postprocessing method "+self.method)
        return ranks

    def _reference(self):
        return "LFPRO fairness \\cite{tsioutsiouliklis2020fairness}" if self.method == "O" else "multiplicative fairness \\cite{tsioutsiouliklis2020fairness}"


class FairWalk(Postprocessor):
    """Adjusting graph convolutions to perform fair random walking [rahman2019fairwalk].."""

    def __init__(self, ranker):
        """ Initializes Fairwalk given a base ranker. **This explicitly assumes immutability** of graphs. If you edit
         graphs also clear the dictionary where preprocessed graphs are inputted by calling *fairwalk.reweights.clear().*

        Args:
            ranker: Optional. The base ranker instance. If None (default), a Tautology() ranker is created.
        """
        super().__init__(ranker)

    def _reweigh(self, graph, sensitive):
        sensitive = to_signal(graph, sensitive)
        if not getattr(self, "reweighs", None):
            self.reweighs = dict()
        if graph not in self.reweighs:
            phi = sum(sensitive.values())/len(graph)
            new_graph = graph.copy()
            for u, v, d in new_graph.edges(data=True):
                d["weight"] = 1./(sensitive[u]*phi+(1-sensitive[u])*(1-phi))
            self.reweighs[graph] = new_graph
        return self.reweighs[graph]

    def rank(self, graph, personalization, sensitive, *args, **kwargs):
        personalization = to_signal(graph, personalization)
        original_graph = personalization.graph
        graph = self._reweigh(original_graph, sensitive)
        personalization = to_signal(graph, dict(personalization.items()))
        ranks = self.ranker.rank(graph, personalization, *args, **kwargs)
        ranks.graph = original_graph
        return ranks

    def transform(self, *args, **kwargs):
        raise Exception("FairWalk can not transform graph signals")

    def _reference(self):
        return "fair random walks \\cite{rahman2019fairwalk}"
