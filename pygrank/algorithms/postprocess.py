import warnings
from pygrank.algorithms.utils import optimize
from math import exp
import random


class Tautology:
    """ Returns ranks as-are.

    This class can be used as a baseline against which to compare other rank post-processing algorithms
    (e.g. those of this package).
    """

    def __init__(self):
        pass

    def transform(self, ranks):
        return ranks

    def rank(self, _, personalization):
        return personalization


class Normalize:
    """ Normalizes ranks by dividing with their maximal value."""

    def __init__(self, ranker=None, method="max"):
        """ Initializes the class with a base ranker instance.

        Attributes:
            ranker: The base ranker instance. A Tautology() ranker is created if None (default) was specified.
            method: Divide ranks either by their "max" (default) or by their "sum"

        Example:
            >>> from pygrank.algorithms.postprocess import Threshold
            >>> G, seed_values, algorithm = ...
            >>> algorithm = Threshold(0.5, algorithm) # sets ranks >= 0.5 to 1 and lower ones to 0
            >>> ranks = algorithm.rank(G, seed_values)

        Example (same outcome, quicker one-time use):
            >>> from pygrank.algorithms.postprocess import Normalize
            >>> G, seed_values, algorithm = ...
            >>> ranks = Normalize(0.5).transform(algorithm.rank(G, seed_values))
        """
        if ranker is not None and not callable(getattr(ranker, "rank", None)):
            ranker, method = method, ranker
            if not callable(getattr(ranker, "rank", None)):
                ranker = None
        self.ranker = Tautology() if ranker is None else ranker
        self.method = method

    def _transform(self, ranks):
        if self.method == "max":
            max_rank = max(ranks.values())
        elif self.method == "sum":
            max_rank = sum(ranks.values())
        else:
            raise Exception("Can only normalize towards max or sum")
        return {node: rank / max_rank for node, rank in ranks.items()}

    def transform(self, ranks, *args, **kwargs):
        return self._transform(self.ranker.transform(ranks, *args, **kwargs))

    def rank(self, G, personalization, *args, **kwargs):
        return self._transform(self.ranker.rank(G, personalization, *args, **kwargs))


class Ordinals:
    """ Converts ranking outcome to ordinal numbers.

    The highest rank is set to 1, the second highest to 2, etc.
    """

    def __init__(self, ranker=None):
        """ Initializes the class with a base ranker instance.

        Attributes:
            ranker: Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified.
        """
        self.ranker = Tautology() if ranker is None else ranker

    def _transform(self, ranks):
        return {v: ord+1 for ord, v in enumerate(sorted(ranks, key=ranks.get, reverse=False))}

    def transform(self, ranks, *args, **kwargs):
        return self._transform(self.ranker.transform(ranks, *args, **kwargs))

    def rank(self, G, personalization, *args, **kwargs):
        return self._transform(self.ranker.rank(G, personalization, *args, **kwargs))


class Threshold:
    """ Converts ranking outcome to binary values based on a threshold value."""

    def __init__(self, threshold="gap", ranker=None):
        """ Initializes the Threshold postprocessing scheme.

        Attributes:
            threshold: Optional. The minimum numeric value required to output rank 1 instead of 0. If "gap" (default)
                then its value is automatically determined based on the maximal percentage increase between consecutive
                ranks.
            ranker: Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified.

        Example:
            >>> from pygrank.algorithms.postprocess import Threshold
            >>> G, seed_values, algorithm = ...
            >>> algorithm = Threshold(0.5, algorithm) # sets ranks >= 0.5 to 1 and lower ones to 0
            >>> ranks = algorithm.rank(G, seed_values)

        Example (same outcome):
            >>> from pygrank.algorithms.postprocess import Threshold
            >>> G, seed_values, algorithm = ...
            >>> ranks = Threshold(0.5).transform(algorithm.rank(G, seed_values))
        """
        if ranker is not None and not callable(getattr(ranker, "rank", None)):
            ranker, threshold = threshold, ranker
            if not callable(getattr(ranker, "rank", None)):
                ranker = None
        self.ranker = Tautology() if ranker is None else ranker
        self.threshold = threshold
        if threshold == "gap":
            warnings.warn("gap-determined threshold is still under development (its implementation may be incorrect)", stacklevel=2)

    def _transform(self, ranks, G):
        threshold = self.threshold
        if threshold == "none":
            return ranks
        if threshold == "gap":
            ranks = {v: ranks[v] / G.degree(v) for v in ranks}
            max_diff = 0
            threshold = 0
            prev_rank = 0
            for v in sorted(ranks, key=ranks.get, reverse=True):
                if prev_rank > 0:
                    diff = (prev_rank - ranks[v]) / prev_rank
                    if diff > max_diff:
                        max_diff = diff
                        threshold = ranks[v]
                prev_rank = ranks[v]
        return {v: 1  if ranks[v] >= threshold else 0 for v in ranks.keys()}

    def transform(self, ranks, *args, **kwargs):
        return self._transform(self.ranker.transform(ranks, *args, **kwargs))

    def rank(self, G, personalization, *args, **kwargs):
        return self._transform(self.ranker.rank(G, personalization, *args, **kwargs))


class Sweep:
    def __init__(self, ranker, uniform_ranker=None):
        self.ranker = ranker
        self.uniform_ranker = ranker if uniform_ranker is None else uniform_ranker

    def rank(self, G, personalization, *args, **kwargs):
        ranks = self.ranker.rank(G, personalization, *args, **kwargs)
        uniforms = self.uniform_ranker.rank(G, {v: 1 for v in G}, *args, **kwargs)
        return {v: ranks[v]/uniforms[v] for v in G}


class FairSweep:
    def __init__(self, ranker, uniform_ranker=None):
        self.ranker = ranker
        self.uniform_ranker = ranker if uniform_ranker is None else uniform_ranker
        warnings.warn("FairSweep is not yet optimize to run with scipy and may be much slower than the ranking algorithm", stacklevel=2)

    def rank(self, G, personalization, sensitive, *args, **kwargs):
        ranks = self.ranker.rank(G, personalization, *args, **kwargs)
        uniforms = self.uniform_ranker.rank(G, {v: 1 for v in G}, *args, **kwargs)
        phi = sum(sensitive.values())/len(ranks)
        sumR = sum(ranks[v] * sensitive.get(v, 0) for v in ranks)
        sumB = sum(ranks[v] * (1 - sensitive.get(v, 0)) for v in ranks)
        numR = sum(sensitive.values())
        numB = len(ranks) - numR
        return {v: ranks[v]/uniforms[v]*((phi*sensitive[v]/sumR)+(1-phi)*(1-sensitive[v])/sumB) for v in G}


class CULEP:
    def __init__(self, ranker):
        self.ranker = ranker

    def __culep(self, ranks, sensitive, params, original_ranks):
        a_sensitive = params[0]
        a_nonsensitive = params[1]
        b_sensitive = params[2]
        b_nonsensitive = params[3]
        a = {v: sensitive.get(v, 0)*a_sensitive+(1-sensitive.get(v, 0))*a_nonsensitive for v in ranks}
        b = {v: sensitive.get(v, 0)*b_sensitive+(1-sensitive.get(v, 0))*b_nonsensitive for v in ranks}
        max_ranks = max(ranks.values())
        max_original_ranks = max(original_ranks.values())
        return {v: ranks[v]*((1-a[v])*exp(b[v]*(ranks[v]/max_ranks-original_ranks.get(v,0)/max_original_ranks))+a[v]*exp(-b[v]*(ranks[v]/max_ranks-original_ranks.get(v,0)/max_original_ranks))) for v in ranks}

    def __prule_loss(self, ranks, original_ranks, sensitive, personalization):
        eps = 1.E-12
        p1 = sum([ranks[v] for v in ranks if sensitive[v] == 0 and v not in personalization]) / (eps+sum([1 for v in ranks if sensitive[v] == 0 and v not in personalization]))
        p2 = sum([ranks[v] for v in ranks if sensitive[v] == 1 and v not in personalization]) / (eps+sum([1 for v in ranks if sensitive[v] == 1 and v not in personalization]))
        p1o = sum([ranks[v] for v in ranks if sensitive[v] == 0]) / (eps + sum([1 for v in ranks if sensitive[v] == 0]))
        p2o = sum([ranks[v] for v in ranks if sensitive[v] == 1]) / (eps + sum([1 for v in ranks if sensitive[v] == 1]))
        max_ranks = max(ranks.values())#/len(ranks)
        max_original_ranks = max(original_ranks.values())#/len(original_ranks)
        return sum(abs(ranks[v]/max_ranks-original_ranks[v]/max_original_ranks)/len(original_ranks) for v in ranks)-min(p1o,p2o)/(eps+max(p1o,p2o))

    def rank(self, G, personalization, sensitive, *args, **kwargs):
        ranks = self.ranker.rank(G, personalization, *args, **kwargs)
        #original_ranks = {v: personalization.get(v,0) for v in G}
        original_ranks = {v: 1 for v in G}
        params = optimize(lambda params: self.__prule_loss(self.__culep(ranks, sensitive, params, original_ranks), ranks, sensitive, personalization), [1, 1, 10, 10], min_vals=[0, 0, -10, -10], tol=1.E-3, divide_range=2, partitions=10)
        #print(params)
        return self.__culep(ranks, sensitive, params, personalization)

class PersonalizationFair:
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


class Fair:
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
