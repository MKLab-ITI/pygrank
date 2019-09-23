import random


def split_groups(groups, fraction_of_training=0.99):
    if fraction_of_training == 1:
        return groups, groups
    clusters = {}
    training = {}
    for group_id, group in groups.items():
        splt = int(len(group)*fraction_of_training)
        # group = list(group) # not really needed if data are already imported as lists
        random.shuffle(group)
        training[group_id] = group[:splt]
        clusters[group_id] = group[splt:]
    return training, clusters


def remove_group_edges_from_graph(G, group):
    for v in group:
        for u in group:
            if G.has_edge(v,u):
                G.remove_edge(v,u)
            if G.has_edge(u, v):
                G.remove_edge(u,v)
