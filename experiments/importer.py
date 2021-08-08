import networkx as nx


def import_SNAP(dataset, path='data/', pair_file='pairs.txt', group_file='groups.txt', directed=False, min_group_size=10, max_group_number=10, import_label_file=True):
    G = nx.DiGraph() if directed else nx.Graph()
    groups = {}
    with open(path+dataset+'/'+pair_file, 'r', encoding='utf-8') as file:
        for line in file:
            if len(line) != 0 and line[0] != '#':
                splt = line[:-1].split('\t')
                if len(splt) == 0:
                    continue
                G.add_edge(splt[0], splt[1])
    if import_label_file:
        with open(path+dataset+'/'+group_file, 'r', encoding='utf-8') as file:
            for line in file:
                if line[0] != '#':
                    group = [item for item in line[:-1].split('\t') if len(item) > 0 and item in G]
                    if len(group) >= min_group_size:
                        groups[len(groups)] = group
                        if len(groups) >= max_group_number:
                            break
    return G, groups


_loaded = dict()


def fairness_dataset(dataset, group=0, sensitive_group=0, path="data/"):
    if dataset not in _loaded:
        if "twitter" in dataset:
            import data.twitter_fairness.importer
            G, sensitive, labels = data.twitter_fairness.importer.load()
        elif "facebook" in dataset:
            import data.facebook_fairness.importer
            if "686" in dataset:
                G, sensitive, labels = data.facebook_fairness.importer.load(path+"facebook_fairness/686")
            else:
                G, sensitive, labels = data.facebook_fairness.importer.load(path+"facebook_fairness/0")
        else:
            G, groups = import_SNAP(dataset, path=path, max_group_number=max(group,sensitive_group)+1, min_group_size=100)
            group = set(groups[group])
            labels = {v: 1 if v in group else 0 for v in G}
            if group == sensitive_group:
                sensitive = {v: 1-labels[v] for v in G}
            else:
                group = set(groups[sensitive_group])
                sensitive = {v: 1 if v in group else 0 for v in G}
            #labels = {v: labels[v] for v in G if labels[v]==1 or sensitive[v]==1}
            #sensitive = {v: sensitive[v] for v in labels}
        _loaded[dataset] = G, sensitive, labels
    return _loaded[dataset]