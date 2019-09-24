import networkx as nx
import metrics.utils
import metrics.unsupervised
import metrics.supervised
import metrics.multigroup
import algorithms.postprocess
import algorithms.pagerank


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
training_groups, test_groups = metrics.utils.split_groups(groups)
metrics.utils.remove_group_edges_from_graph(G, test_groups)

# run algorithms
algorithm = algorithms.postprocess.Normalize(algorithms.pagerank.PageRank())
ranks = {group_id: algorithm.rank(G, {v: 1 for v in group}) for group_id, group in training_groups.items()}

# print Conductance evaluation
metric = metrics.multigroup.MultiUnsupervised(metrics.unsupervised.Conductance, G)
print(metric.evaluate(ranks))

# print AUC evaluation
metric = metrics.multigroup.MultiSupervised(metrics.supervised.AUC, metrics.utils.to_seeds(test_groups))
print(metric.evaluate(ranks))
