import warnings
from pygrank.algorithms.postprocess.postprocess import Tautology, Normalize
from pygrank.algorithms.utils import optimize
from math import exp


def to_fairwalk(G, sensitive):
    eps = 1.E-12
    G = G.to_directed()
    sensitive_sum = {u: 0 for u in G}
    degrees = {u: 0 for u in G}
    for u, v in G.edges():
        sensitive_sum[u] += sensitive.get(v, 0)
        degrees[u] += 1
    for u, v, d in G.edges(data=True):
        d["weight"] = sensitive.get(v, 0) * degrees[u]/(eps+sensitive_sum[u]) + (1 - sensitive.get(v, 0)) * degrees[u]/(eps+degrees[u]-sensitive_sum[u])
    return G


class FairSweep:
    def __init__(self, ranker, uniform_ranker=None):
        self.ranker = ranker
        self.uniform_ranker = ranker if uniform_ranker is None else uniform_ranker

    def rank(self, G, personalization, sensitive, *args, **kwargs):
        ranks = self.ranker.rank(G, personalization, *args, **kwargs)
        uniforms = self.uniform_ranker.rank(G, {v: 1 for v in G}, *args, **kwargs)
        phi = sum(sensitive.values())/len(ranks)
        sumR = sum(ranks[v] * sensitive.get(v, 0) for v in ranks)
        sumB = sum(ranks[v] * (1 - sensitive.get(v, 0)) for v in ranks)
        numR = sum(sensitive.values())
        numB = len(ranks) - numR
        return {v: ranks[v]/uniforms[v]*((phi*sensitive[v]/sumR)+(1-phi)*(1-sensitive[v])/sumB) for v in G}


class FairPersonalizer:
    def __init__(self, ranker, target_pRule=1, retain_rank_weight=1, pRule_weight=1):
        self.ranker = ranker
        self.target_pRule = target_pRule
        self.retain_rank_weight = retain_rank_weight
        self.pRule_weight = pRule_weight

    def __culep(self, personalization, sensitive, ranks, params):
        a = {v: sensitive.get(v, 0)*params[0]+(1-sensitive.get(v, 0))*params[1] for v in ranks}
        b = {v: sensitive.get(v, 0)*params[2]+(1-sensitive.get(v, 0))*params[3] for v in ranks}
        max_ranks = max(ranks.values())
        max_personalization = max(personalization.values())

        return {v: (1 - a[v]) * exp(b[v]*(ranks[v]/max_ranks - personalization.get(v, 0) / max_personalization))
                               + a[v] * exp(-b[v]*(ranks[v] / max_ranks - personalization.get(v, 0) / max_personalization)) for v in ranks}

    def __prule_loss(self, ranks, original_ranks, sensitive, personalization):
        eps = 1.E-12
        p1 = sum([ranks[v] for v in ranks if sensitive[v] == 0]) / (eps + sum([1 for v in ranks if sensitive[v] == 0]))
        p2 = sum([ranks[v] for v in ranks if sensitive[v] == 1]) / (eps + sum([1 for v in ranks if sensitive[v] == 1]))
        max_ranks = max(ranks.values())#/len(ranks)
        max_original_ranks = max(original_ranks.values())#/len(original_ranks)
        return self.retain_rank_weight*sum(abs(ranks[v]/max_ranks-original_ranks[v]/max_original_ranks)/len(original_ranks) for v in ranks)-self.pRule_weight*min(self.target_pRule,min(p1,p2)/(eps+max(p1,p2)))

    def rank(self, G, personalization, sensitive, *args, **kwargs):
        ranks = self.ranker.rank(G, personalization, *args, **kwargs)
        params = optimize(lambda params: self.__prule_loss(self.ranker.rank(G, personalization=self.__culep(personalization, sensitive, ranks, params), *args, **kwargs), ranks, sensitive, personalization), [1, 1, 10, 10], min_vals=[0, 0, -10, -10], tol=1.E-3, divide_range=2, partitions=10)
        return self.ranker.rank(G, personalization=self.__culep(personalization, sensitive, ranks, params), *args, **kwargs)


class FairPostprocessor:
    def __init__(self, ranker=None, method="O"):
        if ranker is not None and not callable(getattr(ranker, "rank", None)):
            ranker, method = method, ranker
            if not callable(getattr(ranker, "rank", None)):
                ranker = None
        self.ranker = Tautology() if ranker is None else ranker
        self.method = method

    def __distribute(self, DR, ranks, sensitive):
        #ranks = {v: ranks[v] * sensitive.get(v, 0) for v in ranks if ranks[v] * sensitive.get(v, 0) > 1.E-6}
        while True:
            ranks = {v: ranks[v] * sensitive.get(v, 0) for v in ranks if ranks[v] * sensitive.get(v, 0) != 0}
            d = DR / len(ranks)
            min_rank = min(val for val in ranks.values())
            if min_rank > d:
                ranks = {v: val - d for v, val in ranks.items()}
                break
            ranks = {v: val - min_rank for v, val in ranks.items()}
            DR -= len(ranks) * min_rank
        return ranks

    def __reweight(self, G, sensitive):
        if not getattr(self, "reweights", None):
            self.reweights = dict()
        if G not in self.reweights:
            phi = sum(sensitive.values())/len(G)
            Gnew = G.copy()
            for u, v, d in Gnew.edges(data=True):
                d["weight"] = 1./(sensitive[u]*phi+(1-sensitive[u])*(1-phi))
            self.reweights[G] = Gnew
        return self.reweights[G]

    def _transform(self, ranks, sensitive):
        phi = sum(sensitive.values())/len(ranks)
        if self.method == "O":
            ranks = Normalize(method="sum").transform(ranks)
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
            raise Exception("Invalid fairness postprocessing method", self.method)
        return ranks

    def transform(self, ranks, sensitive, *args, **kwargs):
        if self.method == "reweight":
            raise Exception("Reweighting can only occur by preprocessing the graph")
        return self._transform(self.ranker.transform(ranks, *args, **kwargs), sensitive)

    def rank(self, G, personalization, sensitive, *args, **kwargs):
        if self.method == "reweight":
            return self.ranker.rank(self.__reweight(G, sensitive), personalization, *args, **kwargs)
        return self._transform(self.ranker.rank(G, personalization, *args, **kwargs), sensitive)
