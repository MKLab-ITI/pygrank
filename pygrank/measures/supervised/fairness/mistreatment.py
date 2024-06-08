from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import to_signal, GraphSignalData, BackendPrimitive
from pygrank.measures.combination import Parity
from typing import Callable
from pygrank.measures.supervised.ranking.auc import AUC


class Mistreatment(Supervised):
    """Computes a disparate mistreatment assessment to test the fairness of given scores given
    that they are similarly evaluated by a measure of choice."""

    def __init__(
        self,
        known_scores: GraphSignalData,
        sensitive: GraphSignalData,
        exclude: GraphSignalData = None,
        measure: Callable[[GraphSignalData, GraphSignalData], Supervised] = AUC,
    ):
        """

        Args:
            sensitive: A binary graph signal that separates sensitive from non-sensitive nodes.
            measure: A supervised measure to compute disparate mistreament on. Default is AUC.

        Example:
            >>> import pygrank as pg
            >>> known_score_signal, sensitive_signal = ...
            >>> train, test = pg.split(known_scores, 0.8)  # 20% test set
            >>> ranker = pg.LFPR()
            >>> measure = pg.Mistreatment(known_scores, exclude=train, measure=pg.AUC)
            >>> scores = ranker(train, sensitive=sensitive_signal)
            >>> print(measure(scores))
        """
        super().__init__(known_scores, exclude)
        self.sensitive = sensitive
        self.measure = measure

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        sensitive = to_signal(scores, self.sensitive)
        if self.exclude is not None:
            exclude = to_signal(sensitive, self.exclude)
            return Parity(
                [
                    self.measure(self.known_scores, 1 - (1 - exclude) * sensitive),
                    self.measure(
                        self.known_scores, 1 - (1 - exclude) * (1 - sensitive)
                    ),
                ]
            ).evaluate(scores)
        else:
            return Parity(
                [
                    self.measure(self.known_scores, None),
                    self.measure(self.known_scores, None),
                ]
            ).evaluate(scores)
