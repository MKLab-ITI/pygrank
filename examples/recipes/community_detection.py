import pygrank as pg


def overlapping_community_detection(graph, known_members, top=None):
    graph_filter = pg.PageRank(0.9) if len(known_members) < 50 else pg.ParameterTuner().tune(graph, known_members)
    ranks = pg.to_signal(graph, {v: 1 for v in known_members}) >> pg.Sweep(graph_filter) >> pg.Normalize("range")
    if top is not None:
        ranks = ranks*(1-pg.to_signal(graph, {v: 1 for v in known_members}))  # set known member scores to zero
        return sorted(list(graph), key=lambda node: -ranks[node])[:top]  # return specific number of top predictions

    threshold = pg.optimize(max_vals=[1], divide_range=1.005, loss=lambda p: pg.Conductance(graph)(pg.Threshold(p[0]).transform(ranks)))[0]
    return [v for v in graph if ranks[v] > threshold]


_, graph, group = next(pg.load_datasets_one_community(["citeseer"]))
train, test = pg.split(group, 3)
found = overlapping_community_detection(graph, train)

# set-based evaluation (instead of working on graph signals)
new = set(found)-set(train)
TP = len(new.intersection(test))
print(f"Precision {TP/len(new):.3f}")
print(f"Recall    {TP/len(test):.3f}")
print(len(found), len(graph))
