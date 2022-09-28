import pygrank as pg
import networkx as nx

graph, features, groups = pg.load_feature_dataset('citeseer', groups_no_labels=True, graph_api=nx)
nodes = list(graph)
# graph = nx.Graph()
for node in nodes:
    graph.add_node(node)
wordnames = {i: "w"+str(i) for i in range(features.shape[1])}
for i, node in enumerate(nodes):
    for j in range(features.shape[1]):
        if features[i, j] != 0:
            graph.add_edge(node, wordnames[j], weight=features[i, j])

group_lists = list(groups.values())
groups = [pg.to_signal(graph, group) for group in groups.values()]
accs = list()
for seed in range(100):
    ranker = pg.PageRank(0.85, renormalize=True, assume_immutability=True,
                         use_quotient=False, error_type="iters", max_iters=10)  # 10 iterations

    #ranker = pg.LowPassRecursiveGraphFilter([1 - .9 / (pg.log(i + 1) + 1) for i in range(10)], renormalize=True, assume_immutability=True, tol=None)

    training, test = pg.split(nodes, 0.8, seed=seed)
    training = set(training)
    ranks_set = [ranker(graph, {node: 1 for node in group if node in training}) for group in group_lists]
    options = list(range(len(ranks_set)))
    found_set = [list() for _ in training]
    tp = 0
    for v in test:
        if max(options, key=lambda i: ranks_set[i][v]) == max(options, key=lambda i: groups[i][v]):
            tp += 1
    accs.append(tp/len(test))
    print(sum(accs)/len(accs))
