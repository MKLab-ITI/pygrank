import networkx as nx
import time
from pygrank.algorithms.pagerank import PageRank
from pygrank.algorithms.pagerank import Fast
from pygrank.algorithms.utils import preprocessor
from scipy.stats import spearmanr


def import_SNAP_data(dataset='',path='data/', pair_file='pairs.txt', group_file='groups.txt', directed=False, min_group_size=1000):
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




normal_time = list()
fast_time = list()
repeats = 1
tol = 1.E-9
alpha = 0.99
errors = list()
correlations = list()
pre = preprocessor('col', assume_immutability=True)
pre(G)  # do this one to make the hash storage not affect anything else

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
    page_rank = PageRank(alpha=alpha, to_scipy=pre, tol=tol, max_iters=500)
    #fast_rank = PageRank(alpha=alpha*0.8, to_scipy=pre, tol=tol, max_iters=500)
    fast_rank = Fast(alpha=alpha, to_scipy=pre, tol=tol, max_iters=500, error_adaptation=0.01)
    seeds = {v:1 for v in groups[0]}
    tic = time.clock()
    ranks_page = page_rank.rank(G, seeds)
    normal_time.append(time.clock() - tic)
    print(page_rank.convergence)
    tic = time.clock()
    ranks_fast = fast_rank.rank(G, seeds)
    fast_time.append(time.clock() - tic)
    print(fast_rank.convergence)
    errors.append(sum(abs(ranks_page[v] - ranks_fast[v]) / len(ranks_page) for v in ranks_page))
    correlations.append(spearmanr(list(ranks_page.values()), list(ranks_fast.values()))[0])
print('Times (normal vs fast)',sum(normal_time), sum(fast_time))
print('Error\t', sum(errors) / len(errors))
print('Spearmanr\t', sum(correlations) / len(correlations))
