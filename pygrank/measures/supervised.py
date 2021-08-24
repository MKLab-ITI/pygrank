import numpy as np
import sklearn.metrics
import scipy
from pygrank.measures.utils import Measure
from pygrank import backend
from pygrank.core.signals import GraphSignal, to_signal
import numbers


class Supervised(Measure):
    """Provides a base class with the ability to simultaneously convert ranks and known ranks to numpy arrays.
    This class is used as a base for other supervised evaluation measures."""

    def __init__(self, known_ranks, exclude=None):
        """
        Initializes the supervised measure with desired graph signal outcomes.
        Args:
            known_ranks: The desired graph signal outcomes.
            exclude: Optional. An iterable (e.g. list, map, networkx graph, graph signal) whose items/keys are traversed
                to determine which nodes to ommit from the evaluation, for example because they were used for training.
                If None (default) the measure is evaluated on all graph nodes. You can safely set the `self.exclude`
                property at any time to alter this original value. Prefer using this behavior to avoid overfitting
                measure assessments.
        """
        self.known_ranks = known_ranks
        self.exclude = exclude

    def to_numpy(self, ranks, normalization=False):
        if isinstance(ranks, numbers.Number) and isinstance(self.known_ranks, numbers.Number):
            return backend.to_array([self.known_ranks]), backend.to_array([ranks])
        elif isinstance(ranks, GraphSignal):
            return to_signal(ranks, self.known_ranks).filter(exclude=self.exclude), ranks.normalized(normalization).filter(exclude=self.exclude)
        elif isinstance(self.known_ranks, GraphSignal):
            return self.known_ranks.filter(exclude=self.exclude), to_signal(self.known_ranks, ranks).normalized(normalization).filter(exclude=self.exclude)
        else:
            if self.exclude is not None:
                raise Exception("Needs to parse graph signal ranks or known_ranks to be able to exclude specific nodes")
            ranks = backend.self_normalize(backend.to_array(ranks, copy_array=True)) if normalization else backend.to_array(ranks)
            return backend.to_array(self.known_ranks), ranks

    def best_direction(self):
        """
        Automatically determines if higher or lower values of the measure are better.
        Design measures so that outcomes of this method depends **only** on their class,
        as it follows a class-based hashing to guarantee speed.

        Returns:
            1 if higher values of the measure are better, -1 otherwise.
        """
        ret = getattr(self.__class__, "__best_direction", None)
        if ret is None:
            ret = 1 if self.__class__([1, 0])([1, 0]) > self.__class__([1, 0])([0, 1]) else -1
            setattr(self.__class__, "__best_direction", ret)
        return ret


class NDCG(Supervised):
    """Provides evaluation of NDCG@k score between given and known ranks."""

    def __init__(self, known_ranks, exclude=None, k=None):
        """ Initializes the PageRank scheme parameters.

        Args:
            k: Optional. Calculates NDCG@k. If None (default), len(known_ranks) is used.
        """
        super().__init__(known_ranks, exclude=exclude)
        if k is not None and k > len(known_ranks):
            raise Exception("NDCG@k cannot be computed for k greater than the number of known ranks")
        self.k = len(known_ranks) if k is None else k

    def evaluate(self, ranks):
        known_ranks, ranks = self.to_numpy(ranks)
        DCG = 0
        IDCG = 0
        for i, v in enumerate(list(sorted(list(range(backend.length(ranks))), key=ranks.__getitem__, reverse=True))[:self.k]):
            DCG += known_ranks[v] / np.log2(i + 2)
        for i, v in enumerate(list(sorted(list(range(backend.length(known_ranks))), key=known_ranks.__getitem__, reverse=True))[:self.k]):
            IDCG += known_ranks[v] / np.log2(i + 2)
        return DCG / IDCG


class MaxDifference(Supervised):
    """Computes the maximum absolute error between ranks and known ranks."""

    def evaluate(self, ranks):
        known_ranks, ranks = self.to_numpy(ranks)
        return backend.max(backend.abs(known_ranks-ranks))


class Mabs(Supervised):
    """Computes the mean absolute error between ranks and known ranks."""

    def evaluate(self, ranks):
        known_ranks, ranks = self.to_numpy(ranks)
        return backend.sum(backend.abs(known_ranks-ranks)) / backend.length(ranks)


class CrossEntropy(Supervised):
    """Computes a cross-entropy loss of given vs known ranks."""

    def evaluate(self, ranks):
        known_ranks, ranks = self.to_numpy(ranks)
        #thresh = backend.min(ranks[known_ranks!=0])
        #ranks = 1/(1+np.exp(-ranks/thresh+1))
        eps = 1.E-14
        ret = -backend.dot(known_ranks, backend.log(ranks+eps))-backend.dot(1-known_ranks, backend.log(1-ranks+eps))
        return ret


class KLDivergence(Supervised):
    """Computes the KL-divergence of given vs known ranks."""

    def evaluate(self, ranks):
        known_ranks, ranks = self.to_numpy(ranks, normalization=True)
        ratio = (ranks+1.E-12)/(known_ranks+1.E-12)
        if backend.min(ratio) <= 0:
            raise Exception("Invalid KLDivergence calculations (negative ranks or known ranks)")
        ret = -backend.dot(ranks, backend.log((known_ranks+1.E-12)/(ranks+1.E-12)))
        #backend.dot(ranks[original_ranks != 0],-backend.log(original_ranks[original_ranks != 0] / ranks[original_ranks != 0]))
        return ret


class AUC(Supervised):
    """Wrapper for sklearn.metrics.auc evaluation."""

    def evaluate(self, ranks):
        known_ranks, ranks = self.to_numpy(ranks)
        if backend.min(known_ranks) == backend.max(known_ranks):
            raise Exception("Cannot evaluate AUC when all labels are the same")
        fpr, tpr, _ = sklearn.metrics.roc_curve(known_ranks, ranks)
        return sklearn.metrics.auc(fpr, tpr)


class Accuracy(Supervised):
    """Computes the accuracy as 1- mean absolute differences between given and known ranks."""

    def evaluate(self, ranks):
        known_ranks, ranks = self.to_numpy(ranks)
        return 1-backend.sum(backend.abs(known_ranks - ranks)) / backend.length(ranks)


class SpearmanCorrelation(Supervised):
    """Computes the Spearman correlation coefficient between given and known ranks."""

    def evaluate(self, ranks):
        known_ranks, ranks = self.to_numpy(ranks)
        return scipy.stats.spearmanr(known_ranks, ranks)[0]


class PearsonCorrelation(Supervised):
    """Computes the Pearson correlation coefficient between given and known ranks."""

    def evaluate(self, ranks):
        known_ranks, ranks = self.to_numpy(ranks)
        return scipy.stats.pearsonr(known_ranks, ranks)[0]


class pRule(Supervised):
    """Computes an assessment of stochastic ranking fairness.
    Values near 1 indicate full fairness, whereas lower values indicate disparate impact.
    Known ranks correspond to whether nodes are sensitive.
    Usually, pRule > 80% is considered fair.
    """

    def evaluate(self, ranks):
        sensitive, ranks = self.to_numpy(ranks)
        p1 = backend.dot(ranks, sensitive)
        p2 = backend.sum(ranks) - p1
        if p1 == 0 or p2 == 0:
            return 0
        s = backend.sum(sensitive)
        p1 /= s
        p2 /= backend.length(sensitive)-s
        return min(p1,p2)/max(p1,p2)
