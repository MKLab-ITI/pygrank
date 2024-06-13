import pygrank as pg
import networkx as nx

graph = next(pg.load_datasets_graph(["citeseer"], graph_api=nx))


def link_prediction(graph, node, top=5):
    algorithm = pg.PageRank(0.95) >> pg.SeedOversampling()
    ranks = algorithm(graph, {node: 1})
    return sorted(graph, key=lambda v: -ranks[v] if not graph.has_edge(node, v) else 0)[
        :top
    ]


precisions = list()
recalls = list()
for node in graph:
    if graph.degree(node) < 10:
        continue
    _, test = pg.split(list(graph.neighbors(node)))
    test = set(test)
    for v in test:
        graph.remove_edge(node, v)
    recommendation = link_prediction(graph, node)
    TP = len([v for v in recommendation if v in test])
    precisions.append(TP / len(recommendation))
    recalls.append(TP / len(test))
    for v in test:
        graph.add_edge(node, v)
print("Avg. precision", sum(precisions) / len(precisions))
print("Avg. recall", sum(recalls) / len(recalls))
