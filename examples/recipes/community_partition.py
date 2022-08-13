import pygrank as pg


def community_detection(graph, known_members_set):
    ranks_set = [pg.ParameterTuner()(graph, known_members) for known_members in known_members_set]
    options = list(range(len(ranks_set)))
    found_set = [list() for _ in known_members_set]
    for v in graph:
        found_set[max(options, key=lambda i:ranks_set[i][v])].append(v)
    return found_set


_, graph, groups = next(pg.load_datasets_multiple_communities(["citeseer"]))
train_set, test_set = pg.split(groups, 0.5)
train_set = train_set.values()
test_set = test_set.values()
found_set = community_detection(graph, train_set)
precisions = list()
recalls = list()
for found, train, test in zip(found_set, train_set, test_set):
    train, test = set(train), set(test)
    new_nodes = [v for v in found if v not in train]
    TP = len([v for v in new_nodes if v in test])
    precisions.append(TP/len(new_nodes) if new_nodes else 0)
    recalls.append(TP/len(test))
print("Avg. precision", sum(precisions)/len(precisions))
print("Avg. recall", sum(recalls)/len(recalls))
