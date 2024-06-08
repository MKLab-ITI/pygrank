from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import backend, GraphSignalData, BackendPrimitive
import scipy.stats


class MannWhitneyParity(Supervised):
    """
    Performs a two-tailed Mann-Whitney U-test to check that the scores of sensitive-attributed nodes (ground truth)
    do not exhibit  higher or lower values compared to the rest. To do this, the test's U statistic is transformed so
    that value 1 indicates that the probability of sensitive-attributed nodes exhibiting higher values is the same as
    for lower values (50%). Value 0 indicates that either the probability of exhibiting only higher or only lower
    values is 100%.
    Known scores correspond to the binary sensitive attribute checking whether nodes are sensitive.
    """

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        sensitive, scores = self.to_numpy(scores)
        scores1 = scores[sensitive == 0]
        scores2 = scores[sensitive != 0]
        return 1 - 2 * abs(
            0.5
            - scipy.stats.mannwhitneyu(scores1, scores2)[0]
            / len(scores1)
            / len(scores2)
        )
