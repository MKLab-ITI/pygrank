import networkx as nx
import os
import gzip
import wget
import shutil
import sys

datasets = {
    "dblp": {"url": "https://snap.stanford.edu/data/com-DBLP.html",
             "pairs": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
             "groups": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.all.cmty.txt.gz"},
    "eucore": {"url": "https://snap.stanford.edu/data/email-Eu-core.html",
               "pairs": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
               "labels": "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz"}
}

def download_dataset(dataset):
    source = datasets[dataset.lower()]
    if not os.path.isdir("data"):
        os.mkdir("data")
    download_path = "data/"+dataset
    if not os.path.isdir(download_path):
        os.mkdir(download_path)
        pairs_path = download_path+"/pairs."+source["pairs"].split(".")[-1]
        wget.download(source["pairs"], pairs_path)
        with gzip.open(pairs_path, 'rb') as f_in:
            with open(download_path+"/pairs.txt", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(pairs_path)

        if "groups" in source:
            groups_path = download_path+"/groups."+source["groups"].split(".")[-1]
            wget.download(source["groups"], groups_path)
            with gzip.open(groups_path, 'rb') as f_in:
                with open(download_path+"/groups.txt", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(groups_path)
        elif "labels" in source:
            labels_path = download_path+"/labels."+source["labels"].split(".")[-1]
            wget.download(source["labels"], labels_path)
            with gzip.open(labels_path, 'rb') as f_in:
                with open(download_path+"/labels.txt", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(labels_path)
            groups = dict()
            with open(download_path+"/labels.txt", 'r', encoding='utf-8') as file:
                for line in file:
                    if line[0] != '#':
                        splt = line[:-1].split()
                        if len(splt) >= 2:
                            if splt[1] not in groups:
                                groups[splt[1]] = list()
                            groups[splt[1]].append(splt[0])
            with open(download_path+"/groups.txt", 'w', encoding='utf-8') as file:
                for group in groups.values():
                    file.write((" ".join(group))+"\n")
    credentials = "Please visit the url "+source["url"]+" for instruction on how to cite the dataset "+dataset+" in your research"
    print(credentials, file=sys.stderr)
    return credentials


def import_SNAP(dataset : str,
                path : str = 'data/',
                pair_file: str = 'pairs.txt',
                group_file: str = 'groups.txt',
                directed: bool = False,
                min_group_size: int = 10,
                max_group_number: int = 20):
    G = nx.DiGraph() if directed else nx.Graph()
    groups = {}
    with open(path+dataset+'/'+pair_file, 'r', encoding='utf-8') as file:
        for line in file:
            if len(line) != 0 and line[0] != '#':
                splt = line[:-1].split()
                if len(splt) != 0:
                    G.add_edge(splt[0], splt[1])
    if group_file is not None:
        with open(path+dataset+'/'+group_file, 'r', encoding='utf-8') as file:
            for line in file:
                if line[0] != '#':
                    group = [item for item in line[:-1].split() if len(item) > 0 and item in G]
                    if len(group) >= min_group_size:
                        groups[len(groups)] = group
                        if len(groups) >= max_group_number:
                            break
    return G, groups


def dataset_loader(datasets=datasets):
    datasets = [(dataset, 0) if len(dataset) != 2 else dataset for dataset in datasets]
    last_loaded_dataset = None
    for dataset, group_id in datasets:
        if last_loaded_dataset != dataset:
            download_dataset(dataset)
            G, groups = import_SNAP(dataset, max_group_number=1+max(group_id for dat, group_id in datasets if dat == dataset))
            last_loaded_dataset = dataset
        group = set(groups[group_id])
        yield dataset, G, group