"""
This file covers the experiments of the paper: Stopping Personalized PageRank without an Error Tolerance Parameter
"""

from pygrank.algorithms import PageRank
from pygrank.algorithms import preprocessor, RankOrderConvergenceManager
from scipy.stats import spearmanr
import numpy as np
import networkx as nx


def import_SNAP_data(dataset, path='data/', pair_file='pairs.txt', group_file='groups.txt', directed=False, min_group_size=10, max_group_number=10, import_label_file=False, specific_ids=None):
    if specific_ids is not None:
        min_group_size = 100000000
        max_group_number = len(specific_ids)
    G = nx.DiGraph() if directed else nx.Graph()
    groups = {}
    with open(path+dataset+'/'+pair_file, 'r', encoding='utf-8') as file:
        for line in file:
            if len(line) != 0 and line[0] != '#':
                splt = line[:-1].split('\t')
                if len(splt) == 0:
                    continue
                G.add_edge(splt[0], splt[1])
    if import_label_file:
        pass

    else:
        with open(path+dataset+'/'+group_file, 'r', encoding='utf-8') as file:
            i = 1
            for line in file:
                if line[0] != '#':
                    group = [item for item in line[:-1].split('\t') if len(item) > 0 and item in G]
                    if len(group) >= min_group_size or i in specific_ids:
                        groups[len(groups)] = group
                        print(i, len(group))
                        if len(groups) >= max_group_number:
                            break
                    i += 1
    return G, groups


def estimate_mixing(alpha):
    sup_sup = -np.log(1 - alpha)
    series_sum = 0
    for n in range(1, 10000):
        series_sum += np.power(alpha, n) / n
        if series_sum > sup_sup * 0.99:
            break
    return int(n)


def show_correlations(ranks, ground_truth):
    from matplotlib import pyplot as plt
    from scipy.stats import rankdata
    ranks = rankdata(ranks)
    ground_truth = rankdata(ground_truth)
    plt.scatter(ground_truth, ranks)
    plt.grid()
    plt.show()


# CHANGE THE FOLLOWING BLOCK TO SELECT DATASET
specific_ids = [1723] # community ids
dataset = 'snap_amazon' # dataset
dataset_name = dataset


G, groups = import_SNAP_data(dataset, specific_ids=specific_ids)#left one is amazon, right is dblp
pre = preprocessor('col', assume_immutability=True) # a preprocessor that hashes the outcome of normalization for faster running time of the same algoriths
pre(G) # run once the preprocessor to not affect potential time measurements

result_spearmans = ""
result_iterations = ""

for group_number in range(len(groups)):
    for alpha in [0.85, 0.90, 0.95, 0.99, 0.995, 0.999]:
        result_spearmans += dataset_name+"-"+str(specific_ids[group_number])+" & & "+(str(alpha)[1:])
        result_iterations += dataset_name+"-"+str(specific_ids[group_number])+" & & "+(str(alpha)[1:])
        seeds = {v:1 for v in groups[group_number]}
        ground_truth_ranker = PageRank(alpha=alpha, to_scipy=pre, tol=1.E-20, max_iters=30000, use_quotient=False)
        ground_truth_ranks = ground_truth_ranker.rank(G, seeds)
        result_iterations += " & "+str(ground_truth_ranker.convergence.iteration)
        print("Found ground truth ranks ("+str(ground_truth_ranker.convergence.iteration)+" iterations)")
        compared_rankers = list()
        for tol in [1.E-6, 1.E-7, 1.E-8, 1.E-9, 1.E-10, 1.E-11, 1.E-12]:
            compared_rankers.append(PageRank(alpha=alpha, to_scipy=pre, tol=tol, max_iters=30000, use_quotient=False))
        compared_rankers.append(PageRank(alpha=alpha, to_scipy=pre, tol=tol, max_iters=estimate_mixing(alpha), error_type="iters"))
        compared_rankers.append(PageRank(alpha=alpha, to_scipy=pre, use_quotient=False, convergence=RankOrderConvergenceManager(alpha, confidence=0.99, criterion="fraction_of_walks")))
        compared_rankers.append(PageRank(alpha=alpha, to_scipy=pre, use_quotient=False, convergence=RankOrderConvergenceManager(alpha, confidence=0.98, criterion="rank_gap")))
        for ranker in compared_rankers:
            ranks = ranker.rank(G, seeds)
            sp = spearmanr(list(ranks.values()), list(ground_truth_ranks.values()))
            #show_correlations(list(ranks.values()), list(ground_truth_ranks.values()))
            #print(sp[0])
            result_spearmans += " & "+str(-int(np.log10(1-sp[0])*10)/10.)
            result_iterations += " & "+str(ranker.convergence.iteration)
            print('----------------------------------')
            print(result_spearmans)
            print(result_iterations)
        result_spearmans += "\\\\\n"
        result_iterations += "\\\\\n"