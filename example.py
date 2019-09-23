import networkx as nx
import metrics.utils
import metrics.unsupervised
import metrics.supervised
import algorithms.utils
import algorithms.pagerank
import algorithms.oversampling


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
for group in test_groups.values():
    metrics.utils.remove_group_edges_from_graph(G, group)

# run algorithms
algorithm = algorithms.pagerank.PageRank()
ranks = {group_id: algorithm.rank(G, {v: 1 for v in group}) for group_id, group in training_groups.items()}

# print Conductance evaluation
ranks = {group_id: algorithms.utils.normalize(group_ranks) for group_id, group_ranks in ranks.items()}
metric = metrics.unsupervised.FastSweep
evaluations = {group_id: metric(G).evaluate(group_ranks) for group_id, group_ranks in ranks.items()}
print(evaluations)

metric = metrics.supervised.AUC
evaluations = {group_id: metric({v: 1 for v in test_groups[group_id]}).evaluate(group_ranks) for group_id, group_ranks in ranks.items()}
print(evaluations)
