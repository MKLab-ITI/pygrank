from pygrank.core import backend, GraphSignalData, BackendPrimitive
from pygrank.measures.utils import Measure
from typing import Iterable, Tuple, Optional
from math import isinf


def _differentiable_hinge(x, gamma=30):
    # doi:10.1088/1742-6596/1743/1/012025, pp. 4
    x = backend.to_primitive(x)
    return x+backend.log(1+backend.exp(-x*gamma))/gamma


class MeasureCombination(Measure):
    """Combines several measures. Measures can be aggregated either by passing them to the constructor or to the
    `add(measure, weight=1, min_val=-infinity, max_val=infinity)` method."""
    
    def __init__(self,
                 measures: Optional[Iterable[Measure]] = None,
                 weights: Optional[Iterable[float]] = None,
                 thresholds: Optional[Iterable[Tuple[float]]] = None,
                 differentiable=False):
        """
        Instantiates a combination of several measures. More measures with their own weights and threhsolded range
        can be added with the `add(measure, weight=1, min_val=-inf, max_val=inf)` method.

        Args:
            measures: Optional. An iterable of measures to combine. If None (default) no new measure is added.
            weights: Optional. A iterable of floats with which to weight the measures provided by the previous
                argument. The concept of weighting depends on how measures are aggregated, but it corresponds
                to an importance value placed on each measure. If None (default), provided measures are all
                weighted by 1.
            thresholds: Optional. A tuple of [min_val, max_val] with which to bound measure outcomes. If None
                (default) provided measures
            differentiable: Optional. If True, a differentiable hinge loss is used to approximate max and min.
                Default is False.

        Example:
            >>> import pygrank as pg
            >>> known_scores, algorithm, personalization, sensitivity_scores = ...
            >>> auc = pg.AUC(known_scores, exclude=personalization)
            >>> prule = pg.pRule(sensitivity_scores, exclude=personalization)
            >>> measure = pg.AM([auc, prule], weights=[1., 10.], thresholds=[(0,1), (0, 0.8)])
            >>> print(measure(algorithm(personalization)))

        Example (same result):
            >>> import pygrank as pg
            >>> known_scores, algorithm, personalization, sensitivity_scores = ...
            >>> auc = pg.AUC(known_scores, exclude=personalization)
            >>> prule = pg.pRule(sensitivity_scores, exclude=personalization)
            >>> measure = pg.AM().add(auc, weight=1., max_val=1).add(prule, weight=1., max_val=0.8)
            >>> print(measure(algorithm(personalization)))
        """
        self.measures = list() if measures is None else measures
        self.weights = [1. for _ in self.measures] if weights is None else weights
        self.thresholds = [(0., 1.) for _ in self.measures] if thresholds is None else thresholds
        self.differentiable = differentiable

    def add(self,
            measure: Measure,
            weight: float = 1.,
            min_val: float = -float('inf'),
            max_val: float = float('inf')):
        self.measures.append(measure)
        self.weights.append(weight)
        self.thresholds.append((min_val, max_val))
        return self

    def _total_weight(self):
        return backend.sum(backend.abs(backend.to_array(self.weights)))

    def max(self, x, constant):
        if self.differentiable and not isinf(constant):
            # TODO: check if this exact expression is mathematically correct (min has been checked)
            return _differentiable_hinge(x-constant)+constant
        return max(x, constant)

    def min(self, x, constant):
        if self.differentiable and not isinf(constant):
            return constant-_differentiable_hinge(constant-x)
        return min(x, constant)


class AM(MeasureCombination):
    """Combines several measures through their arithmetic mean."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        result = 0
        for i in range(len(self.measures)):
            if self.weights[i] != 0:
                measure_evaluation = self.measures[i].evaluate(scores)
                evaluation = self.min(self.max(measure_evaluation, self.thresholds[i][0]), self.thresholds[i][1])
                result += self.weights[i]*evaluation
        return result / self._total_weight()


class Disparity(MeasureCombination):
    """Combines measures by calculating the absolute value of their weighted differences.
    If more than two measures *measures=[M1,M2,M3,M4,...]* are provided this calculates *abs(M1-M2+M3-M4+...)*"""
    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        result = 0
        mult = 1
        for i in range(len(self.measures)):
            if self.weights[i] != 0:
                evaluation = self.measures[i].evaluate(scores)
                evaluation = self.min(self.max(evaluation, self.thresholds[i][0]), self.thresholds[i][1])
                result += (self.weights[i]*mult)*evaluation
            mult *= -1
        return result if result > 0 else -result


class Parity(MeasureCombination):
    """Combines measures by calculating the absolute value of their weighted differences subtracted from 1.
    If more than two measures *measures=[M1,M2,M3,M4,...]* are provided this calculates *1-abs(M1-M2+M3-M4+...)*"""
    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        result = 0
        mult = 1
        for i in range(len(self.measures)):
            if self.weights[i] != 0:
                evaluation = self.measures[i](scores)
                evaluation = self.min(self.max(evaluation, self.thresholds[i][0]), self.thresholds[i][1])
                result += (self.weights[i]*mult)*evaluation
            mult *= -1
        return 1-(result if result > 0 else -result)


class GM(MeasureCombination):
    """Combines several measures through their geometric mean."""
    
    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        result = 0
        for i in range(len(self.measures)):
            if self.weights[i] != 0:
                evaluation = self.measures[i](scores)
                evaluation = self.min(self.max(evaluation, self.thresholds[i][0]), self.thresholds[i][1])
                result += self.weights[i]*backend.log(backend.to_primitive(self.max(backend.epsilon(), evaluation)))
        return backend.exp(result / self._total_weight())
