from pygrank.algorithms.postprocess.postprocess import Tautology, Normalize
from pygrank.algorithms.utils import optimize
from pygrank.algorithms.postprocess import Postprocessor
from math import exp
from pygrank.measures.supervised import pRule
import numpy as np


def to_fairwalk(G, sensitive):
    eps = 1.E-12
    G = G.to_directed()
    sensitive_sum = {u: 0 for u in G}
    degrees = {u: 0 for u in G}
    for u, v in G.edges():
        sensitive_sum[u] += sensitive.get(v, 0)
        degrees[u] += 1
    for u, v, d in G.edges(data=True):
        d["weight"] = (sensitive.get(v, 0) * 1./(eps+sensitive_sum[u]) + (1 - sensitive.get(v, 0)) * 1./(eps+degrees[u]-sensitive_sum[u])) * degrees[u]#/(degrees[u]*degrees[v])**0.5
    return G


class FairWeights(Postprocessor):
    """Weights node scores based on whether they are sensitive, so that the sum of sensitive
    and non-sensitive scores are equal.
    """
    def __init__(self, ranker):
        self.ranker = ranker

    def _transform(self, ranks, sensitive):
        phi = sum(sensitive.values())/len(ranks)
        sumR = sum(ranks[v] * sensitive.get(v, 0) for v in ranks)
        sumB = sum(ranks[v] * (1 - sensitive.get(v, 0)) for v in ranks)
        return {v: ranks[v]*((phi*sensitive[v]/sumR)+(1-phi)*(1-sensitive[v])/sumB) for v in G}


class IterativeFairPersonalizer:
    def __init__(self, ranker, target_pRule=1, retain_rank_weight=1, pRule_weight=1):
        self.ranker = ranker
        self.target_pRule = target_pRule
        self.retain_rank_weight = retain_rank_weight
        self.pRule_weight = pRule_weight

    def rank(self, G, personalization, sensitive, *args, **kwargs):
        original_ranks = self.ranker.rank(G, personalization, *args, **kwargs)
        prev_loss = 0
        for iteration in range(10000):
            ranks = self.ranker.rank(G, personalization, *args, **kwargs)
            max_ranks = max(ranks.values())
            max_original_ranks = max(original_ranks.values())
            ranks = {v: ranks[v]/max_ranks for v in ranks}
            max_ranks = 1
            p1nom = sum([ranks[v] for v in ranks if sensitive[v] == 0])
            p1denom = sum([1. for v in ranks if sensitive[v] == 0])
            p2nom = sum([ranks[v] for v in ranks if sensitive[v] == 1])
            p2denom = sum([1. for v in ranks if sensitive[v] == 1])
            p1 = 0 if p1nom == 0 else p1nom/p1denom
            p2 = 0 if p2nom == 0 else p2nom/p2denom
            prule = min(p1, p2)/(1.E-12+max(p1, p2))
            loss = -min(self.target_pRule, prule)*self.pRule_weight + self.retain_rank_weight*sum((ranks[v]/max_ranks-original_ranks[v]/max_original_ranks)**2/len(original_ranks) for v in ranks)
            derivatives = {v: 0 for v in ranks}
            for v in ranks:
                derivatives[v] -= self.retain_rank_weight/len(original_ranks)*(ranks[v]/max_ranks-original_ranks[v]/max_original_ranks)
                if prule <= self.target_pRule:
                    if p1 < p2:
                        if sensitive[v] == 0 and p1nom !=0 and p2nom != 0:
                            derivatives[v] += self.pRule_weight*p2denom/(p1denom*p2nom)
                        elif sensitive[v] == 1 and p1nom !=0 and p2nom != 0:
                            derivatives[v] += self.pRule_weight*(-p1denom*p1nom)/(p1denom*p2nom)**2
                    elif p1 > p2:
                        if sensitive[v] == 1 and p1nom !=0 and p2nom != 0:
                            derivatives[v] += self.pRule_weight*p1denom/(p2denom*p1nom)
                        elif sensitive[v] == 0 and p1nom !=0 and p2nom != 0:
                            derivatives[v] += self.pRule_weight*(-p2denom*p2nom)/(p2denom*p1nom)**2
            personalization = {v: personalization.get(v,0)+derivatives[v]*0.01 /(self.pRule_weight+self.retain_rank_weight) for v in ranks}
            #print(iteration, loss, prule)
            if abs(loss-prev_loss) < 0.00001:
                break
            prev_loss = loss
        return self.ranker.rank(G, personalization, *args, **kwargs)


class FairPersonalizerSlow:
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
        #eps = 1.E-12
        #p1 = sum([ranks.get(v,0) for v in sensitive if sensitive[v] == 0]) / (eps + sum([1 for v in sensitive if sensitive[v] == 0]))
        #p2 = sum([ranks.get(v,0) for v in sensitive if sensitive[v] == 1]) / (eps + sum([1 for v in sensitive if sensitive[v] == 1]))
        max_ranks = max(ranks.values())#/len(ranks)
        max_original_ranks = max(original_ranks.values())#/len(original_ranks)
        return self.retain_rank_weight*sum(abs(ranks[v]/max_ranks-original_ranks[v]/max_original_ranks)/len(original_ranks) for v in ranks)-self.pRule_weight*min(self.target_pRule,self.pRule(ranks))

    def rank(self, G, personalization, sensitive, *args, **kwargs):
        self.pRule = pRule(sensitive)
        ranks = self.ranker.rank(G, personalization, *args, **kwargs)
        params = optimize(lambda params: self.__prule_loss(self.ranker.rank(G, personalization=self.__culep(personalization, sensitive, ranks, params), *args, **kwargs), ranks, sensitive, personalization), [1, 1, 10, 10], min_vals=[0, 0, -10, -10], tol=1.E-2, divide_range=2)
        return self.ranker.rank(G, personalization=self.__culep(personalization, sensitive, ranks, params), *args, **kwargs)



class FairPersonalizer:
    def __init__(self, ranker, target_pRule=1, retain_rank_weight=1, pRule_weight=1, error_type="KL", parameter_buckets=1, max_residual=1):
        self.ranker = ranker
        self.target_pRule = target_pRule
        self.retain_rank_weight = retain_rank_weight
        self.pRule_weight = pRule_weight
        self.error_type = error_type
        self.parameter_buckets = parameter_buckets
        self.max_residual = max_residual

    def __culep(self, personalization, sensitive, ranks, params):
        res = personalization*params[-1]
        ranks = ranks / ranks.max()
        personalization = personalization / personalization.max()
        for i in range(self.parameter_buckets):
            a = sensitive*(params[0+4*i]-params[1+4*i]) + params[1+4*i]
            b = sensitive*(params[2+4*i]-params[3+4*i]) + params[3+4*i]
            if self.error_type == "mabs":
                res += (1-a)*np.exp(b*(ranks-personalization)) + a*np.exp(-b*(ranks-personalization))
            else:
                res += (1-a)*np.exp(b*np.abs(ranks-personalization)) + a*np.exp(-b*np.abs(ranks-personalization))
        return res

    def __prule_loss(self, ranks, original_ranks, sensitive, personalization):
        if self.error_type == "mabs":
            ranks = ranks / ranks.max()
            original_ranks = original_ranks / original_ranks.max()
            return self.retain_rank_weight * np.abs(ranks - original_ranks).sum() / ranks.size - self.pRule_weight * min(self.target_pRule, self.pRule(ranks))
        if self.error_type == "KL":
            ranks = ranks / ranks.sum()
            original_ranks = original_ranks / original_ranks.sum()
            return self.retain_rank_weight * np.dot(ranks[original_ranks!=0], -np.log(original_ranks[original_ranks!=0]/ranks[original_ranks!=0])) - self.pRule_weight * min(self.target_pRule, self.pRule(ranks))
        raise Exception("Invalid error type")

    def rank(self, G, personalization, sensitive, *args, **kwargs):
        self.pRule = pRule(sensitive, G)
        sensitive, personalization = self.pRule.to_numpy(personalization)
        ranks = self.ranker.rank(G, personalization, *args, as_dict=False, **kwargs)
        def loss(params):
            fair_pers = self.__culep(personalization, sensitive, ranks, params)
            fair_ranks = self.ranker.rank(G, personalization=fair_pers, *args, as_dict=False, **kwargs)
            return self.__prule_loss(fair_ranks, ranks, sensitive, personalization)
        params = optimize(loss, [1, 1, 10, 10]*self.parameter_buckets+[self.max_residual], min_vals=[0, 0, -10, -10]*self.parameter_buckets+[0], tol=1.E-2, divide_range=2, partitions=5)
        return self.ranker.rank(G, personalization=self.__culep(personalization, sensitive, ranks, params), *args, **kwargs)






class FairPostprocessor(Postprocessor):
    """Adjusts node scores so that the sum of sensitive nodes is closer to the sum of non-sensitive ones.
    """

    def __init__(self, ranker=None, method="O"):
        """
        Initializes the fairness-aware postprocessor.

        Args:
            ranker: The base ranking algorithm.
            method: The method with which to adjust weights. If "O" (default) an optimal gradual adjustment is performed.
                If "B" node scores are weighted according to whether the nodes are sensitive, so that
                the sum of sensitive node scores becomes equal to the sum of non-sensitive node scores.
                If "reweight" the graph is pre-processed so that, when possible, walks are equally probable to visit
                sensitive or non-sensitive nodes at non-restarting iterations.
        """
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
            min_rank = min(ranks.values())
            #print(min_rank)
            if min_rank < 1.E-12:
                break
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
