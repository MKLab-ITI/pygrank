import math


def normalize(ranks, method="L1", keep=None):
    if keep is None:
        keep = ranks.keys()
    else:
        keep = [u for u in keep if u in ranks]
    if method=="L1":
        mm = min(ranks[u] for u in keep)
        mx = max(ranks[u] for u in keep)
        return {u: (ranks[u]-mm)/(mx-mm) for u in keep}
    elif method=="L2":
        norm = sum(ranks[u]**2 for u in keep)**0.5
        return {u: ranks[u]/norm for u in keep}
    else:
        raise Exception("Only L1 or L2 normalization supported")


def convert_to_ranks(G, seeds, values=None):
    if seeds is None:
        seeds = G.nodes()
    ranks = {v: 0 for v in G.nodes()}
    for v in seeds:
        ranks[v] = 1 if values is None else values[v]
    return ranks


def assert_binary(prior_ranks):
    for v in prior_ranks.values():
        if v!=0 and v!=1:
            raise Exception('Prior ranks were not binary')


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


class PageRank:
    def __init__(self, normalization='Laplacian', alpha = 0.99, msq_error=0.00001, one_class=False):
        self._alpha = alpha
        self._msq_error = msq_error
        self._one_class = one_class
        self._p_symmetric = 1
        if isinstance(normalization, str):
            if normalization=='RowCol':
                self._p = 0.5
                self._p_symmetric = 2
            elif normalization=='Row':
                self._p = 1
            elif normalization=='Laplacian':
                self._p = 0.5
            elif normalization=='None':
                self._p = None
            else:
                raise Exception("Supported normalization methods are 'Laplacian', 'Row', 'None' and number")
        else:
            self._p = normalization
    def rank(self, G, prior_ranks):
        ranks = prior_ranks
        alpha = self._alpha
        one_class = self._one_class
        if self._p is None:
            degv = {v: len(G.nodes()) for v in G.nodes()}
            degu = degv
        else:
            degv = {v : float(len(list(G.neighbors(v))))**(self._p_symmetric*self._p) for v in G.nodes()}
            degu = {u : float(len(list(G.neighbors(u))))**(self._p_symmetric*(1-self._p)) for u in G.nodes()}
        itters = 0
        msq = 0
        while True:
            current_alpha = alpha#*math.exp(-msq/len(G.nodes())*200)
            if current_alpha>alpha-0.5*self._msq_error**0.5:
                current_alpha = alpha
            # print(itters, current_alpha, msq) # to show fast version vs current_alpha = alpha
            itters += 1
            msq = 0
            next_ranks = {}
            for u in G.nodes():
                rank = sum(ranks[v]/degv[v]/degu[u] for v in G.neighbors(u))
                if one_class and prior_ranks[u]==0:
                    next_ranks[u] = rank
                else:
                    next_ranks[u] = rank*current_alpha + prior_ranks[u]*(1-current_alpha)
                msq += (next_ranks[u]-ranks[u])*(next_ranks[u]-ranks[u])
            ranks = next_ranks
            if msq/len(G.nodes())<self._msq_error and current_alpha==alpha:
                break
        # print('Finished in ', itters)
        return ranks


class OversamplingRank:
    def __init__(self, ranking_algorithm=PageRank(), criterion='Ranks'):
        self._ranking_algorithm = ranking_algorithm
        self._criterion = criterion
    def rank(self, G, prior_ranks):
        assert_binary(prior_ranks)
        if self._criterion=='Ranks':
            #original_priors = prior_ranks
            ranks = self._ranking_algorithm.rank(G, prior_ranks)
            threshold = min(ranks[u] for u in G.nodes if prior_ranks[u]==1)
            prior_ranks = {v: 1 if ranks[v]>=threshold else 0 for v in G.nodes()}
            new_ranks = self._ranking_algorithm.rank(G, prior_ranks)
            new_threshold = min(new_ranks[u] for u in G.nodes if prior_ranks[u]==1)
        elif self._criterion=='Cut':
            ranks = self._ranking_algorithm.rank(G, prior_ranks)
            threshold = min(ranks[u]/G.degree[u] for u in G.nodes if prior_ranks[u]==1)
            prior_ranks = {v: 1 if ranks[v]/G.degree[v]>=threshold else 0 for v in G.nodes()}
        elif self._criterion=='Neighbors':
            prev_ranks = prior_ranks
            prior_ranks = prior_ranks.copy()
            for u in G.nodes():
                if prev_ranks[u]==1:
                    for v in G.neighbors(u):
                        prior_ranks[v] = 1
        else:
            raise Exception("Supported oversampling criterions are 'Ranks', 'Cut' and 'Neighbors'")
        return self._ranking_algorithm.rank(G, prior_ranks)


class BoostingRank:
    def __init__(self, ranking_algorithm=PageRank(), objective='Partial', oversampling_strategy='Previous', weight_error=0.001, max_repetitions=100, verbose=False):
        self._ranking_algorithm = ranking_algorithm
        self._objective = objective
        self._oversampling_strategy = oversampling_strategy
        self._weight_error = weight_error
        self._max_repetitions = max_repetitions
        self.verbose = verbose
    def rank(self, G, prior_ranks):
        r0_N = prior_ranks.copy()
        RN = self._ranking_algorithm.rank(G, r0_N)
        for iteration in range(self._max_repetitions):
            if self._oversampling_strategy=='Previous':
                threshold = min(RN[u] for u in G.nodes if r0_N[u]==1)
            elif self._oversampling_strategy=='Original':
                threshold = min(RN[u] for u in G.nodes if prior_ranks[u]==1)
            else:
                raise Exception("Supported oversampling strategies for oversampling boosting seeds are 'Previous' and 'Original'")
            r0_N = {u: 1 if RN[u]>=threshold else 0 for u in G.nodes()}
            #if sum(r0_N.values())>len(G.nodes())/2:
            #    break
            Rr0_N = self._ranking_algorithm.rank(G, r0_N)
            if self._objective=='Partial':
                a_N = sum(r0_N[u]*Rr0_N[u]**2 for u in G.nodes())/float(sum(Rr0_N[u]**2 for u in G.nodes()))-sum(r0_N[u]*RN[u]*Rr0_N[u] for u in G.nodes())/float(sum(Rr0_N[u]**2 for u in G.nodes()))
            elif self._objective=='Naive':
                a_N = 0.5-0.5*sum(r0_N[u]*RN[u]*Rr0_N[u] for u in G.nodes())/float(sum(Rr0_N[u]**2 for u in G.nodes()))
            else:
                raise Exception("Supported boosting objectives are 'Partial' and 'Naive'")
            if self.verbose:
                print('\t Boosting iteration',iteration,': weight %.4f'%a_N, ' for ',len([u for u in G.nodes() if r0_N[u]==1]),'seeds')
            for u in G.nodes():
                RN[u] += a_N*Rr0_N[u]
            if abs(a_N)<=self._weight_error:
                break
        return RN