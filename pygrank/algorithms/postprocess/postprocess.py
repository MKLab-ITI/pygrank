import warnings
from pygrank.algorithms.utils import MethodHasher, to_signal, NodeRanking, _call, to_signal
from pygrank import backend


class Postprocessor(NodeRanking):
    def __init__(self, ranker=None):
        self.ranker = ranker

    def transform(self, ranks, *args, **kwargs):
        return to_signal(ranks, self._transform(self.ranker.transform(ranks, *args, **kwargs)))

    def rank(self, *args, **kwargs):
        ranks = self.ranker.rank(*args, **kwargs)
        call_transform = lambda **kwargs: self._transform(ranks, **kwargs)
        return to_signal(ranks, _call(call_transform, kwargs))

    def _transform(self, ranks):
        raise Exception("Postprocessor subclasses need to implement a _transform method to call default rank or transform methods")


class Tautology(Postprocessor):
    """ Returns ranks as-are.

    Can be used as a baseline against which to compare other postprocessors.
    """

    def __init__(self, ranker=None):
        self.ranker = ranker

    def transform(self, ranks):
        return ranks

    def rank(self, graph=None, personalization=None, *args, **kwargs):
        if self.ranker is not None:
            return self.ranker.rank(graph, personalization, *args, **kwargs)
        return to_signal(graph, personalization)


class Normalize(Postprocessor):
    """ Normalizes ranks by dividing with their maximal value."""

    def __init__(self, ranker=None, method="max"):
        """ Initializes the class with a base ranker instance. Args are automatically filled in and
        re-ordered if at least one is provided.

        Args:
            ranker: The base ranker instance. A Tautology() ranker is created if None (default) was specified.
            method: Divide ranks either by their "max" (default) or by their "sum" or make the lie in the "range" [0,1]
                by subtracting their mean before diving by their max.

        Example:
            >>> from pygrank.algorithms.postprocess import Normalize
            >>> graph, personalization, algorithm = ...
            >>> algorithm = Normalize(0.5, algorithm) # sets ranks >= 0.5 to 1 and lower ones to 0
            >>> ranks = algorithm.rank(graph, personalization)

        Example (same outcome, simpler one-liner):
            >>> from pygrank.algorithms.postprocess import Normalize
            >>> graph, personalization, algorithm = ...
            >>> ranks = Normalize(0.5).transform(algorithm.rank(graph, personalization))
        """
        if ranker is not None and not callable(getattr(ranker, "rank", None)):
            ranker, method = method, ranker
            if not callable(getattr(ranker, "rank", None)):
                ranker = None
        self.ranker = Tautology() if ranker is None else ranker
        self.method = method

    def _transform(self, ranks):
        min_rank = 0
        if self.method == "range":
            max_rank = float(backend.max(ranks.np))
            min_rank = float(backend.min(ranks.np))
        elif self.method == "max":
            max_rank = float(backend.max(ranks.np))
        elif self.method == "sum":
            max_rank = float(backend.sum(ranks.np))
        else:
            raise Exception("Can only normalize towards max or sum")
        if min_rank == max_rank:
            return ranks
        ret = (ranks.np-min_rank) / (max_rank-min_rank)
        return ret


class Ordinals(Postprocessor):
    """ Converts ranking outcome to ordinal numbers.

    The highest rank is set to 1, the second highest to 2, etc.
    """

    def __init__(self, ranker=None):
        """ Initializes the class with a base ranker instance.

        Args:
            ranker: Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified.
        """
        self.ranker = Tautology() if ranker is None else ranker

    def _transform(self, ranks):
        return {v: ord+1 for ord, v in enumerate(sorted(ranks, key=ranks.get, reverse=True))}


class Transformer(Postprocessor):
    """Applies an element-by-element transformation on a graph signal based on a given expression."""

    def __init__(self, ranker=None, expr=backend.exp):
        """ Initializes the class with a base ranker instance. Args are automatically filled in and
        re-ordered if at least one is provided.

        Args:
            ranker: Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified.
            expr: Optional. A lambda expression to apply on each element. The transformer will automatically try to
                apply it on the backend array representation of the graph signal first, so prefer use of backend functions
                for faster computations. For example, backend.exp (default) should be prefered instead of math.exp, because
                the former can directly parse a numpy array.

        Example:
            >>> from pygrank.algorithms.postprocess import Normalize, Transformer
            >>> from pygrank import backend
            >>> graph, personalization, algorithm = ...
            >>> r1 = Normalize(algorithm, "sum").rank(graph, personalization)
            >>> r2 = Transformer(algorithm, lambda x: x/backend.sum(x)).rank(graph, personalization)
            >>> print(sum(abs(r1[v]-r2[v]) for v in graph))
            0
        """
        if ranker is not None and not callable(getattr(ranker, "rank", None)):
            ranker, expr = expr, ranker
            if not callable(getattr(ranker, "rank", None)):
                ranker = None
        self.ranker = Tautology() if ranker is None else ranker
        self.expr = expr

    def _transform(self, ranks):
        try:
            return self.expr(ranks.np)
        except:
            return {v: self.expr(ranks[v]) for v in ranks}


class Threshold(Postprocessor):
    """ Converts ranking outcome to binary values based on a threshold value."""

    def __init__(self, threshold="gap", ranker=None):
        """ Initializes the Threshold postprocessing scheme. Args are automatically filled in and
        re-ordered if at least one is provided.

        Args:
            threshold: Optional. The minimum numeric value required to output rank 1 instead of 0. If "gap" (default)
                then its value is automatically determined based on the maximal percentage increase between consecutive
                ranks.
            ranker: Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified.

        Example:
            >>> from pygrank.algorithms.postprocess import Threshold
            >>> graph, personalization, algorithm = ...
            >>> algorithm = Threshold(0.5, algorithm) # sets ranks >= 0.5 to 1 and lower ones to 0
            >>> ranks = algorithm.rank(graph, personalization)

        Example (same outcome):
            >>> from pygrank.algorithms.postprocess import Threshold
            >>> graph, personalization, algorithm = ...
            >>> ranks = Threshold(0.5).transform(algorithm.rank(graph, personalization))
        """
        if ranker is not None and not callable(getattr(ranker, "rank", None)):
            ranker, threshold = threshold, ranker
            if not callable(getattr(ranker, "rank", None)):
                ranker = None
        self.ranker = Tautology() if ranker is None else ranker
        self.threshold = threshold
        if threshold == "gap":
            warnings.warn("gap-determined threshold is still under development (its implementation may be incorrect)", stacklevel=2)

    def _transform(self, ranks, graph):
        threshold = self.threshold
        if threshold == "none":
            return ranks
        if threshold == "gap":
            ranks = {v: ranks[v] / graph.degree(v) for v in ranks}
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


class Sweep(Postprocessor):
    """
    Applies a sweep procedure that divides personalized node ranks by corresponding non-personalized ones.
    """
    def __init__(self, ranker, uniform_ranker=None):
        """
        Initializes the sweep procedure.

        Args:
            ranker: The base ranker instance.
            uniform_ranker: Optional. The ranker instance used to perform non-personalized ranking. If None (default)
                the base ranker is used.
        """
        self.ranker = ranker
        self.uniform_ranker = ranker if uniform_ranker is None else uniform_ranker
        self.centrality = MethodHasher(lambda graph: self.uniform_ranker.rank(graph), assume_immutability=True)

    def _transform(self, ranks):
        uniforms = self.centrality(ranks.G).np
        return ranks.np/(1.E-12+uniforms.np)
