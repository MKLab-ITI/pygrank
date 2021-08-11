import networkx as nx


def import_SNAP(dataset : str,
                path : str = 'data/',
                pair_file: str = 'pairs.txt',
                group_file: str = 'groups.txt',
                directed: bool = False,
                min_group_size: int = 10,
                max_group_number: int = 10):
    G = nx.DiGraph() if directed else nx.Graph()
    groups = {}
    with open(path+dataset+'/'+pair_file, 'r', encoding='utf-8') as file:
        for line in file:
            if len(line) != 0 and line[0] != '#':
                splt = line[:-1].split('\t')
                if len(splt) == 0:
                    continue
                G.add_edge(splt[0], splt[1])
    if group_file is not None:
        with open(path+dataset+'/'+group_file, 'r', encoding='utf-8') as file:
            for line in file:
                if line[0] != '#':
                    group = [item for item in line[:-1].split('\t') if len(item) > 0 and item in G]
                    if len(group) >= min_group_size:
                        groups[len(groups)] = group
                        if len(groups) >= max_group_number:
                            break
    return G, groups



def dataset_loader(datasets):
    datasets = [(dataset, 0) if len(dataset) != 2 else dataset for dataset in datasets]
    last_loaded_dataset = None
    for dataset, group_id in datasets:
        if last_loaded_dataset != dataset:
            G, groups = import_SNAP(dataset, max_group_number=1+max(group_id for dat, group_id in datasets if dat == dataset))
            last_loaded_dataset = dataset
        group = set(groups[group_id])
        yield dataset, G, group