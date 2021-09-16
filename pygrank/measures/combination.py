from pygrank import backend
from pygrank.measures.utils import Measure
from typing import Iterable, Tuple
from math import log, exp


class MeasureCombination(Measure):
    """Combines several measures. Measures can be aggregated either by passing them to the constructor or to the
    `add(measure, weight=1, min_val=-infinity, max_val=infinity)` method."""
    
    def __init__(self,
                 measures: Iterable[Measure] = None,
                 weights: Iterable[float] = None,
                 thresholds: Iterable[Tuple[float]] = None):
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
        """
        self.measures = list() if measures is None else measures
        self.weights = [1 for _ in self.measures] if weights is None else weights
        self.thresholds = [(0, 1) for _ in self.measures] if thresholds is None else thresholds

    def add(self,
            measure: Measure,
            weight: float = 1.,
            min_val: float = -float('inf'),
            max_val: float = float('inf')):
        self.measures.append(measure)
        self.weights.append(weight)
        self.thresholds.append((min_val, max_val))
        return self


class AM(MeasureCombination):
    """Combines several measures through their arithmetic mean."""

    def evaluate(self, ranks):
        result = 0
        for i in range(len(self.measures)):
            if self.weights[i] != 0:
                eval = self.measures[i].evaluate(ranks)
                result += self.weights[i]*min(max(eval, self.thresholds[i][0]), self.thresholds[i][1])
        return result/sum(self.weights)


class GM(MeasureCombination):
    """Combines several measures through their geometric mean."""
    
    def evaluate(self, ranks):
        result = 0
        for i in range(len(self.measures)):
            if self.weights[i] != 0:
                eval = self.measures[i].evaluate(ranks)
                result += self.weights[i]*log(min(max(eval, self.thresholds[i][0]), max(backend.epsilon(), self.thresholds[i][1])))
        return exp(result/sum(self.weights))
