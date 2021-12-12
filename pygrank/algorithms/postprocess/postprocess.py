from pygrank.algorithms.utils import MethodHasher, call, ensure_used_args, remove_used_args
from pygrank.core.signals import GraphSignal, to_signal, NodeRanking
from pygrank.core import backend, GraphSignalGraph, GraphSignalData
from typing import Union


class Postprocessor(NodeRanking):
    def __init__(self, ranker: NodeRanking = None):
        self.ranker = ranker

    def transform(self, ranks: GraphSignal, *args, **kwargs):
        return to_signal(ranks, call(self._transform, kwargs, [ranks]))

    def rank(self, *args, **kwargs):
        ranks = self.ranker.rank(*args, **kwargs)
        kwargs = remove_used_args(self.ranker.rank, kwargs)
        return to_signal(ranks, call(self._transform, kwargs, [ranks]))

    def _transform(self, ranks: GraphSignal, **kwargs):
        raise Exception("_transform method not implemented for the class "+self.__class__.__name__)

    def _reference(self):
        return self.__class__.__name__

    def references(self):
        if self.ranker is None:
            return [self._reference()]
        refs = self.ranker.references()
        ref = self._reference()
        if ref is not None and len(ref) > 0:
            refs.append(ref)
        return refs


class Tautology(Postprocessor):
    """ Returns ranks as-are.

    Can be used as a baseline against which to compare other postprocessors or graph filters.
    """

    def __init__(self, ranker: NodeRanking = None):
        """Initializes the Tautology postprocessor with a base ranker.

        Args:
            ranker: The base ranker instance. If None (default), this works as a base ranker that returns
             a copy of personalization signals as-are or a conversion of backend primitives into signals.
        """
        super().__init__(ranker)

    def transform(self, ranks: GraphSignal, *args, **kwargs) -> GraphSignal:
        return ranks

    def rank(self,
             graph: GraphSignalGraph = None,
             personalization: GraphSignalData = None,
             *args, **kwargs) -> GraphSignal:
        if self.ranker is not None:
            return self.ranker.rank(graph, personalization, *args, **kwargs)
        return to_signal(graph, personalization)

    def _reference(self):
        return "tautology" if self.ranker is None else ""


class MabsMaintain(Postprocessor):
    """Forces node ranking posteriors to have the same mean absolute value as prior inputs."""

    def __init__(self, ranker):
        """ Initializes the postprocessor with a base ranker instance.

        Args:
            ranker: Optional. The base ranker instance. If None (default), a Tautology() ranker is created.
        """
        super().__init__(Tautology() if ranker is None else ranker)

    def rank(self, graph=None, personalization=None, *args, **kwargs):
        personalization = to_signal(graph, personalization)
        norm = backend.sum(backend.abs(personalization.np))
        ranks = self.ranker(graph, personalization, *args, **kwargs)
        if norm != 0:
            ranks.np = ranks.np * norm / backend.sum(backend.abs(ranks.np))
        return ranks

    def _reference(self):
        return "mabs preservation"


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
            >>> import pygrank as pg
            >>> graph, personalization, algorithm = ...
            >>> algorithm = pg.Normalize(0.5, algorithm) # sets ranks >= 0.5 to 1 and lower ones to 0
            >>> ranks = algorithm.rank(graph, personalization)

        Example (same outcome, simpler one-liner):
            >>> import pygrank as pg
            >>> graph, personalization, algorithm = ...
            >>> ranks = pg.Normalize(0.5).transform(algorithm.rank(graph, personalization))
        """
        if ranker is not None and not callable(getattr(ranker, "rank", None)):
            ranker, method = method, ranker
            if not callable(getattr(ranker, "rank", None)):
                ranker = None
        super().__init__(Tautology() if ranker is None else ranker)
        self.method = method

    def _transform(self, ranks: GraphSignal, **kwargs):
        ensure_used_args(kwargs)
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

    def _reference(self):
        if self.method == "range":
            return "[0,1] " + self.method + " normalization"
        return self.method+" normalization"


class Ordinals(Postprocessor):
    """ Converts ranking outcome to ordinal numbers.

    The highest rank is set to 1, the second highest to 2, etc.
    """

    def __init__(self, ranker=None):
        """ Initializes the class with a base ranker instance.

        Args:
            ranker: Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified.

        Example:
            >>> import pygrank as pg
            >>> graph, personalization, algorithm = ...
            >>> algorithm = pg.Ordinals(algorithm)
            >>> ranks = algorithm.rank(graph, personalization)

        Example (same outcome, simpler one-liner):
            >>> import pygrank as pg
            >>> graph, personalization, algorithm = ...
            >>> ranks = pg.Ordinals().transform(algorithm.rank(graph, personalization))
        """
        super().__init__(Tautology() if ranker is None else ranker)

    def _transform(self, ranks: GraphSignal, **kwargs):
        ensure_used_args(kwargs)
        return {v: order+1 for order, v in enumerate(sorted(ranks, key=ranks.get, reverse=True))}

    def _reference(self):
        return "ordinal conversion"


class Transformer(Postprocessor):
    """Applies an element-by-element transformation on a graph signal based on a given expression."""

    def __init__(self, ranker=None, expr=backend.exp):
        """ Initializes the class with a base ranker instance. Args are automatically filled in and
        re-ordered if at least one is provided.

        Args:
            ranker: Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified.
            expr: Optional. A lambda expression to apply on each element. The transformer will automatically try to
                apply it on the backend array representation of the graph signal first, so prefer pygrank's backend
                functions for faster computations. For example, backend.exp (default) should be preferred instead of
                math.exp, because the former can directly parse numpy arrays, tensors, etc.

        Example:
            >>> import pygrank as pg
            >>> graph, personalization, algorithm = ...
            >>> r1 = pg.Normalize(algorithm, "sum").rank(graph, personalization)
            >>> r2 = pg.Transformer(algorithm, lambda x: x/pg.sum(x)).rank(graph, personalization)
            >>> print(pg.Mabs(r1)(r2))
        """
        if ranker is not None and not callable(getattr(ranker, "rank", None)):
            ranker, expr = expr, ranker
            if not callable(getattr(ranker, "rank", None)):
                ranker = None
        super().__init__(Tautology() if ranker is None else ranker)
        self.expr = expr

    def _transform(self, ranks: GraphSignal, **kwargs):
        ensure_used_args(kwargs)
        try:
            return self.expr(ranks.np)
        except:
            return {v: self.expr(ranks[v]) for v in ranks}

    def _reference(self):
        return "element-by-element "+self.expr.__name__


class Threshold(Postprocessor):
    """ Converts ranking outcome to binary values based on a threshold value."""

    def __init__(self,
                 ranker: Union[str, float, NodeRanking] = None,
                 threshold: Union[str, float, NodeRanking] = "gap"):
        """ Initializes the Threshold postprocessing scheme. Args are automatically filled in and
        re-ordered if at least one is provided.

        Args:
            ranker: Optional. The base ranker instance. A Tautology() ranker is created if None (default) was specified.
            threshold: Optional. The minimum numeric value required to output rank 1 instead of 0. If "gap" (default)
                then its value is automatically determined based on the maximal percentage increase between consecutive
                ranks.

        Example:
            >>> import pygrank as pg
            >>> graph, personalization, algorithm = ...
            >>> algorithm = pg.Threshold(algorithm, 0.5) # sets ranks >= 0.5 to 1 and lower ones to 0
            >>> ranks = algorithm.rank(graph, personalization)

        Example (same outcome):
            >>> import pygrank as pg
            >>> graph, personalization, algorithm = ...
            >>> ranks = pg.Threshold(0.5).transform(algorithm.rank(graph, personalization))
        """
        if ranker is not None and not callable(getattr(ranker, "rank", None)):
            ranker, threshold = threshold, ranker
            if not callable(getattr(ranker, "rank", None)):
                ranker = None
        super().__init__(Tautology() if ranker is None else ranker)
        self.threshold = threshold

    def _transform(self,
                   ranks: GraphSignal,
                   **kwargs):
        ensure_used_args(kwargs)
        threshold = self.threshold
        if threshold == "gap":
            # TODO maybe enable ranks = {v: ranks[v] / ranks.graph.degree(v) for v in ranks} with a postprocessor
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
        return {v: 1 if ranks[v] >= threshold else 0 for v in ranks.keys()}

    def _reference(self):
        return str(self.threshold)+" threshold"


class Sweep(Postprocessor):
    """
    Applies a sweep procedure that divides personalized node ranks by corresponding non-personalized ones.
    """
    def __init__(self,
                 ranker: NodeRanking,
                 uniform_ranker: NodeRanking = None):
        """
        Initializes the sweep procedure.

        Args:
            ranker: The base ranker instance.
            uniform_ranker: Optional. The ranker instance used to perform non-personalized ranking. If None (default)
                the base ranker is used.

        Example:
            >>> import pygrank as pg
            >>> graph, personalization, algorithm = ...
            >>> algorithm = pg.Sweep(algorithm) # divides node scores by uniform ranker'personalization non-personalized outcome
            >>> ranks = algorithm.rank(graph, personalization

        Example with different rankers:
            >>> import pygrank as pg
            >>> graph, personalization, algorithm, uniform_ranker = ...
            >>> algorithm = pg.Sweep(algorithm, uniform_ranker=uniform_ranker)
            >>> ranks = algorithm.rank(graph, personalization)

        Example (same outcome):
            >>> import pygrank as pg
            >>> graph, personalization, uniform_ranker, algorithm = ...
            >>> ranks = pg.Threshold(uniform_ranker).transform(algorithm.rank(graph, personalization))
        """
        super().__init__(ranker)
        self.uniform_ranker = ranker if uniform_ranker is None else uniform_ranker
        self.centrality = MethodHasher(lambda graph: self.uniform_ranker.rank(graph), assume_immutability=True)

    def _transform(self,
                   ranks: GraphSignal,
                   **kwargs):
        ensure_used_args(kwargs)
        uniforms = self.centrality(ranks.graph).np
        return ranks.np/(1.E-12+uniforms)

    def _reference(self):
        return "sweep ratio postprocessing \\cite{andersen2007local}"  # TODO: allow references to account for custom centrality ranker
