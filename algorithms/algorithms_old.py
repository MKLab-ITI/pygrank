import math
import networkx as nx
import scipy

"""
DEPRECATED - THIS FILE WILL GRADUALLY REMOVED IN FUTURE VERSIONS AS PROJECT ORGANIZATION IS IMPROVED
"""

class Selective:
    def __init__(self, candidates, nodes_with_known_labels, measure="AUC", verbose=False):
        self.candidates = candidates
        self._measure = measure
        self._nodes_with_known_labels = nodes_with_known_labels
        self.verbose = verbose

    def rank(self, G, prior_ranks):
        import random
        best_performance = None
        best_rank_alg = None
        #use half the seeds for training
        training_seeds = [u for u in prior_ranks.keys() if prior_ranks[u]==1]
        random.shuffle(training_seeds)
        training_seeds = training_seeds[:len(training_seeds)//2]
        #user known labels without the training seeds for evaluation
        evaluation_nodes = [u for u in self._nodes_with_known_labels if not u in training_seeds]
        training_priors = {u: 1 if u in training_seeds else 0 for u in prior_ranks.keys()}
        for algorithmid, algorithm in enumerate(self.candidates):
            algorithm.verbose = self.verbose
            ranks = algorithm.rank(G, training_priors)
            performance = self.measure({u: ranks[u] for u in evaluation_nodes}, {u: (1-training_priors[u])*prior_ranks[u] for u in evaluation_nodes})
            if self.verbose:
                print('Algorithm',algorithmid,self._measure,':',performance)
            if performance is None:
                performance = -float("inf")
            if best_performance is None or performance>best_performance:
                best_performance = performance
                best_rank_alg = algorithm
        if self.verbose:
            print('Calculating Ranks for best algorithm')
        return best_rank_alg.rank(G, prior_ranks)

    def measure(self, ranks, prior_ranks):
        if self._measure=="AUC":
            from sklearn.metrics import roc_curve, auc
            fpr, tpr, _ = roc_curve(list(prior_ranks.values()), list(ranks.values()))
            return auc(fpr, tpr)
        elif self._measure=="NDCG":
            DCG = 0
            IDCG = 0
            i = 1
            for v in sorted(ranks, key=ranks.get, reverse=True):
                if prior_ranks[v]==1:
                    DCG += 1/math.log2(i+1)
                i += 1
            i = 1
            for v in prior_ranks.keys():
                if prior_ranks[v]==1:
                    IDCG += 1/math.log2(i+1)
                    i += 1
            if IDCG==0:
                return 0
            return DCG/IDCG
        else:
            return "Only AUC or NDCG supported for measure selection"
