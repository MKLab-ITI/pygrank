import warnings


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

    def transform(self, ranks):
        if not isinstance(self.ranker, Tautology):
            raise Exception("transform(ranks) only makes sense for Tautology base ranker. Consider using rank(G, personalization) instead.")
        return self._transform(ranks)

    def rank(self, G, *args, **kwargs):
        ranks = self.ranker.rank(G, *args, **kwargs)
        return self._transform(ranks)


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

    def transform(self, ranks):
        if not isinstance(self.ranker, Tautology):
            raise Exception("transform(ranks) only makes sense for Tautology base ranker. Consider using rank(G, personalization) instead.")
        return self._transform(ranks)

    def rank(self, G, *args, **kwargs):
        ranks = self.ranker.rank(G, *args, **kwargs)
        return self._transform(ranks)


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
        self.ranker = Tautology() if ranker is None else ranker
        self.threshold = threshold
        if threshold == "gap":
            warnings.warn("gap-determined threshold is still under development (its implementation may be incorrect)", stacklevel=2)

    def _transform(self, ranks, G):
        threshold = self.threshold
        if threshold == "none":
            return ranks
        if threshold == "gap":
            #ranks = {v: ranks[v] / G.degree(v) for v in ranks}
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

    def transform(self, ranks):
        if not isinstance(self.ranker, Tautology):
            raise Exception("transform(ranks) only makes sense for Tautology base ranker. Consider using rank(G, personalization) instead.")
        return self._transform(ranks)

    def rank(self, G, *args, **kwargs):
        ranks = self.ranker.rank(G, *args, **kwargs)
        return self._transform(ranks, G)


class Fair:
    def __init__(self, ranker, method="N"):
        self.ranker = ranker
        self.method = method

    def rank(self, G, personalization, sensitive,  **kwargs):
        if self.method == "none":
            return self.ranker.rank(G, personalization, **kwargs)
        G = G.to_directed()
        phi = 1-sum(sensitive.values()) / len(sensitive)
        d0 = {}
        d1 = {}
        num0 = len([u for u in G if sensitive[u] == 0])
        num1 = len([u for u in G if sensitive[u] == 0])
        for v in G:
            sum0 = len([u for u in G.successors(v) if sensitive[u] == 0])
            sum1 = len([u for u in G.successors(v) if sensitive[u] == 1])
            if self.method == "equal":
                for u in list(G.successors(v)):
                    G.add_edge(v, u, weight=phi / sum0 if sensitive[u] == 0 else (1 - phi) / sum1)
            elif self.method == "N":
                if sum0 == 0 and sensitive[v] == 0:
                    for u in G:
                        if sensitive[u] != 0:
                            G.add_edge(v, u, weight=phi / num0)
                elif sum1 == 0 and sensitive[v] == 1:
                    for u in G:
                        if sensitive[u] != 0:
                            G.add_edge(v, u, weight=(1-phi) / num1)
                else:
                    for u in list(G.successors(v)):
                        G.add_edge(v, u, weight=phi / sum0 if sensitive[u] == 0 else (1 - phi) / sum1)
            elif self.method == "P":
                if (1 - phi)*sum0 < phi*sum1:
                    for u in list(G.successors(v)):
                        G.add_edge(v, u, weight=(1 - phi)/sum1)
                    d0[v] = phi - (1 - phi) * sum0 / sum1
                else:
                    for u in list(G.successors(v)):
                        G.add_edge(v, u, weight=phi/sum0)
                    d1[v] = phi - (1 - phi) * sum1 / sum0
        personalization = {v: (1 - phi)*personalization[v]*(1-sensitive[v])+phi*personalization[v]*sensitive[v] for v in personalization}
        return self.ranker.rank(G, personalization, fairness_residuals=[d0, d1], **kwargs)
