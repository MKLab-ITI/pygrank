import numpy as np
import sklearn.metrics
import scipy
from pygrank.measures.utils import Measure
from pygrank.core import backend, GraphSignal, to_signal, GraphSignalData, BackendPrimitive
import numbers
from typing import Tuple, Union


class Time(Measure):
    """An abstract class that can be passed to benchmark experiments to indicate that they should report running time
    of algorithms. Instances of this class have no functionality."""

    pass


class Supervised(Measure):
    """Provides a base class with the ability to simultaneously convert scores and known scores to numpy arrays.
    This class is used as a base for other supervised evaluation measures."""

    def __init__(self, known_scores: GraphSignalData, exclude: GraphSignalData = None):
        """
        Initializes the supervised measure with desired graph signal outcomes.
        Args:
            known_scores: The desired graph signal outcomes.
            exclude: Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed
                to determine which nodes to ommit from the evaluation, for example because they were used for training.
                If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude`
                property at any time to alter this original value. Prefer using this behavior to avoid overfitting
                measure assessments.
        """
        self.known_scores = known_scores
        self.exclude = exclude

    def to_numpy(self, scores: GraphSignalData, normalization: bool = False) -> Union[Tuple[GraphSignal, GraphSignal], Tuple[BackendPrimitive, BackendPrimitive]]:
        if isinstance(scores, numbers.Number) and isinstance(self.known_scores, numbers.Number):
            return backend.to_array([self.known_scores]), backend.to_array([scores])
        elif isinstance(scores, GraphSignal):
            return to_signal(scores, self.known_scores).filter(exclude=self.exclude), scores.normalized(normalization).filter(exclude=self.exclude)
        elif isinstance(self.known_scores, GraphSignal):
            return self.known_scores.filter(exclude=self.exclude), to_signal(self.known_scores, scores).normalized(normalization).filter(exclude=self.exclude)
        else:
            if self.exclude is not None:
                raise Exception("Needs to parse graph signal scores or known_scores to be able to exclude specific nodes")
            scores = backend.self_normalize(backend.to_array(scores, copy_array=True)) if normalization else backend.to_array(scores)
            return backend.to_array(self.known_scores), scores

    def best_direction(self) -> int:
        """
        Automatically determines if higher or lower values of the measure are better.
        Design measures so that outcomes of this method depends **only** on their class,
        as it follows a class-based hashing to guarantee speed. Otherwise override th

        Returns:
            1 if higher values of the measure are better, -1 otherwise.
        """
        ret = getattr(self.__class__, "__best_direction", None)
        if ret is None:
            ret = 1 if self.__class__([1, 0])([1, 0]) > self.__class__([1, 0])([0, 1]) else -1
            setattr(self.__class__, "__best_direction", ret)
        return ret


class NDCG(Supervised):
    """Provides evaluation of NDCG@k score between given and known scores."""

    def __init__(self, known_scores: GraphSignalData, exclude: GraphSignalData = None, k: int =None):
        """ Initializes the supervised measure with desired graph signal outcomes and the number of top scores.

        Args:
            k: Optional. Calculates NDCG@k. If None (default), len(known_scores) is used.
        """
        super().__init__(known_scores, exclude=exclude)
        if k is not None and k > len(known_scores):
            raise Exception("NDCG@k cannot be computed for k greater than the number of known scores")
        self.k = len(known_scores) if k is None else k

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        DCG = 0
        IDCG = 0
        for i, v in enumerate(list(sorted(list(range(backend.length(scores))), key=scores.__getitem__, reverse=True))[:self.k]):
            DCG += known_scores[v] / np.log2(i + 2)
        for i, v in enumerate(list(sorted(list(range(backend.length(known_scores))), key=known_scores.__getitem__, reverse=True))[:self.k]):
            IDCG += known_scores[v] / np.log2(i + 2)
        return DCG / IDCG


class MaxDifference(Supervised):
    """Computes the maximum absolute error between scores and known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        return backend.max(backend.abs(known_scores-scores))


class Mabs(Supervised):
    """Computes the mean absolute error between scores and known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        return backend.sum(backend.abs(known_scores-scores)) / backend.length(scores)


class CrossEntropy(Supervised):
    """Computes a cross-entropy loss of given vs known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        #thresh = backend.min(scores[known_scores!=0])
        #scores = 1/(1+np.exp(-scores/thresh+1))
        eps = backend.epsilon()
        ret = -backend.dot(known_scores, backend.log(scores+eps))-backend.dot(1-known_scores, backend.log(1-scores+eps))
        return ret


class KLDivergence(Supervised):
    """Computes the KL-divergence of given vs known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores, normalization=True)
        eps = backend.epsilon()
        known_scores = known_scores - backend.min(known_scores) + eps
        known_scores = known_scores / backend.sum(known_scores)
        scores = scores - backend.min(scores) + eps
        scores = scores / backend.sum(scores)
        ratio = scores/known_scores
        ret = -backend.sum(scores*backend.log(ratio))
        return ret


class MKLDivergence(Supervised):
    """Computes the KL-divergence of given vs known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores, normalization=True)
        eps = backend.epsilon()
        known_scores = known_scores - backend.min(known_scores) + eps
        known_scores = known_scores / backend.sum(known_scores)
        scores = scores - backend.min(scores) + eps
        scores = scores / backend.sum(scores)
        ratio = scores/known_scores
        ret = -backend.sum(scores*backend.log(ratio))
        return ret/backend.length(scores)


class Cos(Supervised):
    """Computes the cosine similarity between given and known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        divide = backend.dot(known_scores, known_scores) * backend.dot(scores, scores)
        return backend.safe_div(backend.dot(known_scores, scores), divide**0.5)


class Dot(Supervised):
    """Computes the dot similarity between given and known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        return backend.dot(known_scores, scores)


class TPR(Supervised):
    """Computes the true positive rate."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        known_scores = backend.safe_div(known_scores, backend.max(known_scores))
        scores = backend.safe_div(scores, backend.max(scores))
        return backend.safe_div(backend.sum(known_scores*scores), backend.sum(scores))


class TNR(Supervised):
    """Computes the false negative rate."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        known_scores = backend.safe_div(known_scores, backend.max(known_scores))
        scores = backend.safe_div(scores, backend.max(scores))
        return backend.safe_div(backend.sum((1-known_scores)*(1-scores)), backend.sum(1-scores))


class AUC(Supervised):
    """Wrapper for sklearn.metrics.auc evaluation."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        if backend.min(known_scores) == backend.max(known_scores):
            raise Exception("Cannot evaluate AUC when all labels are the same")
        fpr, tpr, _ = sklearn.metrics.roc_curve(known_scores, scores)
        return sklearn.metrics.auc(fpr, tpr)


class Accuracy(Supervised):
    """Computes the accuracy as 1- mean absolute differences between given and known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        return 1-backend.sum(backend.abs(known_scores - scores)) / backend.length(scores)


class SpearmanCorrelation(Supervised):
    """Computes the Spearman correlation coefficient between given and known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        return scipy.stats.spearmanr(known_scores, scores)[0]


class PearsonCorrelation(Supervised):
    """Computes the Pearson correlation coefficient between given and known scores."""

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        known_scores, scores = self.to_numpy(scores)
        return scipy.stats.pearsonr(known_scores, scores)[0]


class pRule(Supervised):
    """Computes an assessment of stochastic ranking fairness by obtaining the fractional comparison of average scores
    between sensitive-attributed nodes and the rest the rest.
    Values near 1 indicate full fairness (statistical parity), whereas lower values indicate disparate impact.
    Known scores correspond to the binary sensitive attribute checking whether nodes are sensitive.
    Usually, pRule > 80% is considered fair.
    """

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        sensitive, scores = self.to_numpy(scores)
        p1 = backend.dot(scores, sensitive)
        p2 = backend.sum(scores) - p1
        if p1 == 0 or p2 == 0:
            return 0
        s = backend.sum(sensitive)
        p1 = backend.safe_div(p1, s)
        p2 = backend.safe_div(p2, backend.length(sensitive)-s)
        if p1 <= p2:  # this implementation is derivable
            return p1 / p2
        return p2 / p1


class MannWhitneyParity(Supervised):
    """
    Performs a two-tailed Mann-Whitney U-test to check that the scores of sensitive-attributed nodes do not exhibit
    higher or lower values compared to the rest. To do this, the test's U statistic is transformed so that value
    1 indicates that the probability of sensitive-attributed nodes exhibiting higher values is the same as
    for lower values (50%). Value 0 indicates that either the probability of exhibiting only higher or only lower
    values is 100%.
    Known scores correspond to the binary sensitive attribute checking whether nodes are sensitive.
    """

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        sensitive, scores = self.to_numpy(scores)
        scores1 = scores[sensitive == 0]
        scores2 = scores[sensitive != 0]
        return 1-2*abs(0.5-scipy.stats.mannwhitneyu(scores1, scores2)[0]/len(scores1)/len(scores2))
