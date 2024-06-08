from pygrank.measures.supervised.supervised import Supervised
from pygrank.core import backend, GraphSignalData, BackendPrimitive


class L2Disparity(Supervised):
    def __init__(self, *args, target_pRule=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_pRule = target_pRule

    def evaluate(self, scores: GraphSignalData) -> BackendPrimitive:
        sensitive, scores = self.to_numpy(scores)
        p1 = backend.dot(scores, sensitive)
        p2 = backend.dot(scores, 1 - sensitive)
        s = backend.sum(sensitive)
        n = backend.length(sensitive)
        p1 = backend.safe_div(p1, s / float(n))
        p2 = backend.safe_div(p2, 1.0 - s / float(n))
        # if p1 <= p2*self.target_pRule:
        #    p2 *= self.target_pRule
        # elif p2 <= p1*self.target_pRule:
        #    p1 *= self.target_pRule
        # else:
        #    return 0
        return backend.abs(p1 - p2) ** 2
