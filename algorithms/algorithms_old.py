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


class BiasedKernel:
    def __init__(self, normalization='Laplacian', alpha = 0.99, msq_error=0.00001):
        self._alpha = alpha
        self._msq_error = msq_error
        if isinstance(normalization, str):
            if normalization=='Row':
                self._p = 1
            elif normalization=='Laplacian':
                self._p = 0.5
            else:
                raise Exception("Supported normalization methods are 'Laplacian', 'Row' and number")
        else:
            self._p = normalization

    def rank(self, G, prior_ranks):
        ranks = prior_ranks
        degv = {v : float(len(list(G.neighbors(v))))**self._p for v in G.nodes()}
        degu = {u : float(len(list(G.neighbors(u))))**(1-self._p) for u in G.nodes()}
        k = 1
        t = 5
        while True:
            msq = 0
            next_ranks = {}
            for u in G.nodes():
                rank = sum(ranks[v]/degv[v]/degu[u] for v in G.neighbors(u))
                a = self._alpha*t/k
                next_ranks[u] = math.exp(-t)*prior_ranks[u] + a*(rank - ranks[u])
                msq += (next_ranks[u]-ranks[u])*(next_ranks[u]-ranks[u])
            ranks = next_ranks
            k += 1
            if msq/len(G.nodes())<self._msq_error:
                break
        return ranks
    
    
class HeatKernel:
    def __init__(self, normalization='Laplacian', alpha = 0.99, msq_error=0.00001, t=5):
        self._alpha = alpha
        self._msq_error = msq_error
        self._t = t
        if isinstance(normalization, str):
            if normalization=='Row':
                self._p = 1
            elif normalization=='Laplacian':
                self._p = 0.5
            else:
                raise Exception("Supported normalization methods are 'Laplacian', 'Row' and number")
        else:
            self._p = normalization

    def rank(self, G, prior_ranks):
        ranks = prior_ranks
        degv = {v : float(len(list(G.neighbors(v))))**self._p for v in G.nodes()}
        degu = {u : float(len(list(G.neighbors(u))))**(1-self._p) for u in G.nodes()}
        k = 1
        t = self._t
        a = self._alpha*math.exp(-t)
        sum_ranks = {u: ranks[u]*math.exp(-t) for u in G.nodes()}
        while True:
            a = a*t/k
            msq = 0
            next_ranks = {}
            for u in G.nodes():
                rank = sum(ranks[v]/degv[v]/degu[u] for v in G.neighbors(u))
                next_ranks[u] = rank
                sum_ranks[u] += a*next_ranks[u]
                msq += (next_ranks[u]-ranks[u])*(next_ranks[u]-ranks[u])
            ranks = next_ranks
            #print(msq/len(G.nodes())*a)
            k += 1
            if msq/len(G.nodes())*a<self._msq_error:
                break
        return ranks


class Tautology:
    def __init__(self):
        pass
    def rank(self, G, prior_ranks):
        return prior_ranks
