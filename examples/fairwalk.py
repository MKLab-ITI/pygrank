from pygrank.algorithms import *
from pygrank.measures import *
from experiments.importer import fairness_dataset
from pygrank.algorithms import preprocessor
import random
import time


def perc(num):
    if num < 0.005:
        return "0"
    ret = str(int(num*100+.5)/100.)  # helper method to pretty print percentages
    if len(ret) < 4:
        ret += "0"
    if ret[0] == "0":
        return ret[1:]
    return ret


#datasets = ["acm", "amazon", "ant", "citeseer","dblp","facebook0","facebook686","log4j","maven","pubmed","squirel", "twitter"]
datasets = ["facebook0"]
#datasets = ["facebook0","facebook686","pubmed","squirel","twitter"]
seed_fractions = [0.1, 0.2, 0.3]
pre = preprocessor(assume_immutability=True, normalization="symmetric")
#pre = preprocessor(assume_immutability=False, normalization="symmetric")# UNCOMMENT WHEN RUNNING FAIRWALK

graph_filters = {
    "ppr0.85": PageRank(alpha=0.85, to_scipy=pre, max_iters=1000000, tol=1.E-6, assume_immutability=True),
    #"ppr0.99": PageRank(alpha=0.99, preprocessor=pre, max_iters=1000000, tol=1.E-6, assume_immutability=True),
    #"hk3": HeatKernel(t=3, preprocessor=pre, max_iters=1000000, tol=1.E-9, assume_immutability=True),
    #"hk7": HeatKernel(t=7, preprocessor=pre, max_iters=1000000, tol=1.E-9, assume_immutability=True),
}
for filter in list(graph_filters.keys()):
    graph_filters["sweep "+filter] = Sweep(graph_filters[filter])
    del graph_filters[filter]

for filter, H in graph_filters.items():
    print("=====", filter, "=====")
    algorithms = {
        "None": lambda G, p, s: Normalize(H).rank(G, p),
        #"AUCPers": lambda G,p,personalization: personalizer(H, G, p, personalization, 0, 0),
        "FairWalk": lambda G,p,s: Normalize(AdHocFairness(H, "fairwalk")).rank(G, p, sensitive=s),
        "Mult": lambda G,p,s: Normalize(AdHocFairness(H, "B")).rank(G, p, sensitive=s),
        "LFRPO": lambda G,p,s: Normalize(AdHocFairness(H, "O")).rank(G, p, sensitive=s),
        "FairPers": lambda G,p,s: Normalize(FairPersonalizer(H, error_type=Mabs, max_residual=0)).rank(G, p, sensitive=s),
        "FairPers-C": lambda G,p,s: Normalize(FairPersonalizer(H, .80, pRule_weight=10, error_type=Mabs, max_residual=0)).rank(G, p, sensitive=s),
        "FairPersKL": lambda G,p,s: Normalize(FairPersonalizer(H, max_residual=0)).rank(G, p, sensitive=s),
        "FairPersKL-C": lambda G,p,s: Normalize(FairPersonalizer(H,.80, pRule_weight=10, max_residual=0)).rank(G, p, sensitive=s),
      }
    for dataset in datasets:
        random.seed(1) # ensure reproducibility
        G, sensitive, labels = fairness_dataset(dataset, 0, sensitive_group=1, path="../data/")
        #G = to_fairwalk(G, sensitive) # COMMENT THIS LINE WHEN NOT RUNNING EXPLICITLY FAIRWALK AND USE preprocessor(assume_immutability=False, normalization="none")
        #print('Dataset pRule', pRule(sensitive)(labels), 'nodes', len(G), 'edges', G.number_of_edges(), 'positive', sum(labels.values()), 'sensitive', sum(sensitive.values()))
        measures = {"AUC": AUC(labels),
                    "pRule": pRule(sensitive)}
        print_algs = ""
        hash_seeds = dict()
        for algorithm in algorithms:
            measure_scores = {measure: list() for measure in measures}
            for seed_seed, seeds in enumerate(seed_fractions):
                if seed_seed in hash_seeds:
                    personalization, training, evaluation = hash_seeds[seed_seed]
                else:
                    random.seed(seed_seed) # ensure reproducibility'
                    if seeds < 1:
                        training, evaluation = split(list(labels.keys()), training_samples=seeds)
                    else:
                        training_pos = [v for v in labels if labels[v] == 1 and sensitive[v] == 0]
                        random.shuffle(training_pos)
                        training_neg = [v for v in labels if labels[v] == 0 and sensitive[v] == 0]
                        random.shuffle(training_neg)
                        training = training_pos[:seeds]+training_neg[:seeds]
                        evaluation = list(set(labels.keys())-set(training))
                    personalization = {v: labels[v] for v in training if sensitive[v] == 0}
                    hash_seeds[seed_seed] = (personalization, training, evaluation)
                ranks = algorithms[algorithm](G, personalization, sensitive)
                for measure in measures:
                    measures[measure].exclude = training
                    measure_scores[measure].append(measures[measure](ranks))
            print_algs += " & "+" & ".join([str(perc(sum(measure_scores[measure])/len(measure_scores[measure]))) for measure in measure_scores])
        print(dataset[0].upper()+dataset[1:], print_algs, "\\\\")