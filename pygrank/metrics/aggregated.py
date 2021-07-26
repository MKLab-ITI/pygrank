import numpy as np
from .utils import Measure


class __Aggregated__(Measure):
    def __init__(self, metrics=None, weights=None, thresholds=None):
        self.metrics = list() if metrics is None else metrics
        self.weights = [1 for _ in self.metrics] if weights is None else weights
        self.thresholds = [(0,1) for _ in self.metrics] if thresholds is None else thresholds

    def add(self, metric, weight=1, min_val=-float('inf'), max_val=float('inf')):
        self.metrics.append(metric)
        self.weights.append(weight)
        self.thresholds.append((min_val, max_val))
        return self


class AMtarget(__Aggregated__):
    def evaluate(self, ranks):
        result = 0
        for i in range(len(self.metrics)):
            if self.weights[i] != 0:
                eval = self.metrics[i].evaluate(ranks)
                #print("metric", i, ":", eval, "->", min(max(eval, self.thresholds[i][0]), self.thresholds[i][1]))
                result += self.weights[i]*(self.thresholds[i][0]-self.thresholds[i][1])**2
        return result


class AM(__Aggregated__):
    def evaluate(self, ranks):
        result = 0
        for i in range(len(self.metrics)):
            if self.weights[i] != 0:
                eval = self.metrics[i].evaluate(ranks)
                #print("metric", i, ":", eval, "->", min(max(eval, self.thresholds[i][0]), self.thresholds[i][1]))
                result += self.weights[i]*min(max(eval, self.thresholds[i][0]), self.thresholds[i][1])
        return result


class GM(__Aggregated__):
    def evaluate(self, ranks):
        result = 0
        for i in range(len(self.metrics)):
            if self.weights[i] != 0:
                eval = self.metrics[i].evaluate(ranks)
                result += self.weights[i]*np.log(1.E-12+min(max(eval, self.thresholds[i][0]), self.thresholds[i][1]))
        return np.exp(result/sum(self.weights))