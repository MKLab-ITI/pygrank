import pygrank as pg


def overlapping_community_detection(graph, known_members, top=None):
    graph_filter = pg.PageRank(0.9) if len(known_members) < 50 else pg.ParameterTuner().tune(graph, known_members)
    ranks = pg.to_signal(graph, {v: 1 for v in known_members}) >> pg.Sweep(graph_filter) >> pg.Normalize("range")
    if top is not None:
        ranks = ranks*(1-pg.to_signal(graph, {v: 1 for v in known_members}))  # set known member scores to zero
        return sorted(list(graph), key=lambda node: -ranks[node])[:top]  # return specific number of top predictions

    threshold = pg.optimize(max_vals=[1], loss=lambda p: pg.Conductance(graph)(pg.Threshold(p[0]).transform(ranks)))[0]
    known_members = set(known_members)
    return [v for v in graph if ranks[v] > threshold and v not in known_members]


_, graph, group = next(pg.load_datasets_one_community(["citeseer"]))
print(len(group))
train, test = pg.split(group, 0.1)
found = overlapping_community_detection(graph, train)

# node-based evaluation (we work on the returned list of nodes instead of graph signals)
test = set(test)
TP = len([v for v in found if v in test])
print("Precision", TP/len(found))
print("Recall", TP/len(test))
print("Match size", len(found)/len(test))
