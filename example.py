import networkx as nx
import time
from pygrank.algorithms.pagerank import PageRank
from pygrank.algorithms.utils import preprocessor
from scipy.stats import spearmanr
import numpy as np


def import_SNAP_data(dataset='',path='data/', pair_file='pairs.txt', group_file='groups.txt', directed=False, min_group_size=10):
    G = nx.DiGraph() if directed else nx.Graph()
    groups = {}
    with open(path+dataset+'/'+pair_file, 'r', encoding='utf-8') as file:
        for line in file:
            if len(line) != 0 and line[0] != '#':
                splt = line[:-1].split('\t')
                if len(splt) == 0:
                    continue
                G.add_edge(splt[0], splt[1])
    with open(path+dataset+'/'+group_file, 'r', encoding='utf-8') as file:
        for line in file:
            if line[0] != '#':
                group = [item for item in line[:-1].split('\t') if len(item) > 0 and item in G]
                if len(group) >= min_group_size:
                    groups[len(groups)] = group
                    break
    return G, groups


G, groups = import_SNAP_data('youtube')

tol = 1.E-20
alpha = 0.99
errors = list()
pre = preprocessor('col', assume_immutability=True)
pre(G)  # do this one to make the hash storage not affect anything else

def evaluate(fast_rank, page_rank, group_id=0):
    normal_time = list()
    fast_time = list()
    correlations = list()
    repeats = 1

    """ # SHOW RUNNING TIMES
    alpha = 0.1
    page_rank = PageRank(alpha=alpha, to_scipy=pre, tol=tol, max_iters=500)
    for _ in range(50):
        page_rank = PageRank(alpha=alpha, to_scipy=pre, tol=tol, max_iters=500)
        page_rank.rank(G, {v:1 for v in groups[0]})
        print(alpha, page_rank.convergence.iteration)
        alpha = alpha*0.9+1*0.1
    """

    for _ in range(repeats):
        #fast_rank = PageRank(alpha=alpha*0.8, to_scipy=pre, tol=tol, max_iters=500)
        seeds = {v:1 for v in groups[group_id]}
        tic = time.clock()
        ranks_page = page_rank.rank(G, seeds)
        normal_time.append(time.clock() - tic)
        print(page_rank.convergence)
        tic = time.clock()
        ranks_fast = fast_rank.rank(G, seeds)
        fast_time.append(time.clock() - tic)
        print(fast_rank.convergence)
        errors.append(sum(abs(ranks_page[v] - ranks_fast[v]) / len(ranks_page) for v in ranks_page))
        sp = spearmanr(list(ranks_page.values()), list(ranks_fast.values()))
        correlations.append(sp[0])
    print('Times (normal vs fast)',sum(normal_time), sum(fast_time))
    print('Error\t', sum(errors) / len(errors))
    print('Spearmanr\t', sum(correlations) / len(correlations))
    print('logSpearmanr\t', -np.log10(1-sum(correlations) / len(correlations)))

    import matplotlib.pyplot as plt
    from scipy.stats import rankdata
    plt.scatter(rankdata(list(ranks_page.values())), rankdata(list(ranks_fast.values())), color='b')
    plt.show()


page = PageRank(alpha=alpha, to_scipy=pre, tol=tol, max_iters=10000)
#fast = PageRank(alpha=alpha, to_scipy=pre, tol=1.E-8, max_iters=2000)
fast = PageRank(alpha=alpha, to_scipy=pre, tol=tol, max_iters=int(2./(1-alpha)), error_type="iters")
#fast = PageRank(alpha=alpha*0.95, to_scipy=pre, tol=tol, max_iters=2000, error_type="iters")
evaluate(fast, page)