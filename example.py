import networkx as nx
import pygrank.metrics.multigroup
import pygrank.metrics.utils
import pygrank.metrics.unsupervised
import pygrank.metrics.supervised
import pygrank.algorithms
import pygrank.algorithms.postprocess
import pygrank.algorithms.pagerank


def import_SNAP_data(pair_file='data/pairs.txt', group_file='data/groups.txt', directed=False, min_group_size=10):
    G = nx.DiGraph() if directed else nx.Graph()
    groups = {}
    with open(pair_file, 'r', encoding='utf-8') as file:
        for line in file:
            if len(line) != 0 and line[0] != '#':
                splt = line[:-1].split('\t')
                if len(splt) == 0:
                    continue
                G.add_edge(splt[0], splt[1])
    with open(group_file, 'r', encoding='utf-8') as file:
        for line in file:
            if line[0] != '#':
                group = [item for item in line[:-1].split('\t') if len(item) > 0 and item in G]
                if len(group) >= min_group_size:
                    groups[len(groups)] = group
    return G, groups


# setting up experiment data
G, groups = import_SNAP_data()
print(len(groups), "groups", 6000)
training_groups, test_groups = pygrank.metrics.utils.split_groups(groups)
pygrank.metrics.utils.remove_group_edges_from_graph(G, test_groups)

# run algorithms
algorithm = pygrank.algorithms.postprocess.Normalize(pygrank.algorithms.pagerank.Fast(pygrank.algorithms.pagerank.PageRank(alpha=0.99)))
ranks = {group_id: algorithm.rank(G, {v: 1 for v in group}) for group_id, group in training_groups.items()}

# print Conductance evaluation
metric = pygrank.metrics.multigroup.MultiUnsupervised(pygrank.metrics.unsupervised.Conductance, G)
print(metric.evaluate(ranks))

# print LinkAUC evaluation
metric = pygrank.metrics.multigroup.LinkAUC(G, pygrank.metrics.utils.to_nodes(test_groups))
print(metric.evaluate(ranks))

# print AUC evaluation
metric = pygrank.metrics.multigroup.MultiSupervised(pygrank.metrics.supervised.AUC, pygrank.metrics.utils.to_seeds(test_groups))
print(metric.evaluate(ranks))
