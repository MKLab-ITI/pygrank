import gzip
import tarfile
import wget
import shutil
import sys
import os
from pygrank.benchmarks import dataset_processors

datasets = {
    "dblp": {"url": "https://snap.stanford.edu/data/com-DBLP.html",
             "pairs": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
             "groups": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.all.cmty.txt.gz"},
    "eucore": {"url": "https://snap.stanford.edu/data/email-Eu-core.html",
               "pairs": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
               "labels": "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz"},
    "citeseer": {"url": "https://linqs.soe.ucsc.edu/data",
                 "all": "https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz",
                 "pairs": "citeseer/citeseer.cites",
                 "features": "citeseer/citeseer.content",
                 "remove": "citeseer/"},
    "maven": {"url": "https://zenodo.org/record/1489120",
              "all": "https://zenodo.org/record/1489120/files/maven-data.csv.tar.xz",
              "pairs": "maven-data.csv/links_all.csv",
              "remove": "maven-data.csv/",
              "pair_process": lambda row: None if len(row)>1 else [row[0].split(",")[0], row[0].split(",")[1]],
              "node2group": lambda node: node.split(":")[0],
              },
    "amazon": {"url": "https://snap.stanford.edu/data/amazon-meta.html",
               "all": "https://snap.stanford.edu/data/bigdata/amazon/amazon-meta.txt.gz",
               "script": dataset_processors.amazon_processor},
    "graph5": {"url": "https://github.com/maniospas/pygrank-datasets",
               "pairs": "https://raw.githubusercontent.com/maniospas/pygrank-datasets/main/graph5/pairs.txt",
               },
    "graph9": {"url": "https://github.com/maniospas/pygrank-datasets",
               "pairs": "https://raw.githubusercontent.com/maniospas/pygrank-datasets/main/graph9/pairs.txt",
               "groups": "https://raw.githubusercontent.com/maniospas/pygrank-datasets/main/graph9/groups.txt",
               },
    "bigraph": {"url": "https://github.com/maniospas/pygrank-datasets",
                "pairs": "https://raw.githubusercontent.com/maniospas/pygrank-datasets/main/biblock/pairs.txt",
                "groups": "https://raw.githubusercontent.com/maniospas/pygrank-datasets/main/biblock/groups.txt",
                },
    "blockmodel": {"url": "https://github.com/maniospas/pygrank-datasets",
                   "pairs": "https://raw.githubusercontent.com/maniospas/pygrank-datasets/main/blockmodel/pairs.txt",
                   "groups": "https://raw.githubusercontent.com/maniospas/pygrank-datasets/main/blockmodel/groups.txt",
                   },
    "synthfeats": {"url": "https://github.com/maniospas/pygrank-datasets",
                   "pairs": "https://raw.githubusercontent.com/maniospas/pygrank-datasets/main/synthfeats/pairs.txt",
                   "groups": "https://raw.githubusercontent.com/maniospas/pygrank-datasets/main/synthfeats/groups.txt",
                   "features": "https://raw.githubusercontent.com/maniospas/pygrank-datasets/main/synthfeats/features.txt",
                   },
    "pubmed": {"url": "https://linqs.soe.ucsc.edu/data",
               "all": "https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz",
               "pairs": "Pubmed-Diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab",
               "pair_process": lambda row: None if len(row)<4 else [row[0].split(":")[-1], row[3].split(":")[-1]],
               "features": "Pubmed-Diabetes/data/Pubmed-Diabetes.NODE.paper.tab",
               "feature_process": lambda row: None if len(row)<3 or "cat=" in row[0] else [row[0]]+[val for val in row[2:] if "summary=" not in val]+[row[1].split("=")[-1]]
               },
    "cora": {"url": "https://www.dgl.ai/",
             "all": "https://data.dgl.ai/dataset/cora_v2.zip"
             },
}


def downloadable_datasets():
    return list(datasets.keys())


def downloadable_small_datasets():
    return ["bigraph", "blockmodel", "citeseer", "eucore", "graph5", "graph9"]


def download_dataset(dataset, path: str = "data"):   # pragma: no cover
    dataset = dataset.lower()
    if dataset not in datasets:
        return
    source = datasets[dataset] if isinstance(dataset, str) else dataset
    credentials = "REQUIRED CITATION: Please visit the url "+source["url"]\
                  + " for instructions on how to cite the dataset "+dataset+" in your research"
    print(credentials, file=sys.stderr)
    if not os.path.isdir(path):
        os.mkdir(path)
    download_path = path+"/"+dataset
    if not os.path.isdir(download_path):
        os.mkdir(download_path)
        if "all" in source:
            all_path = download_path+"/all."+source["all"].split(".")[-1]
            wget.download(source["all"], all_path)
            try:
                tarfile.open(all_path, 'r').extractall(download_path+"/")
            except tarfile.ReadError:
                with gzip.open(all_path, 'rb') as f_in:
                    with open(download_path+"/all.txt", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            os.remove(all_path)
        if "script" in source:
            source["script"](path)

        if "pairs" in source:
            if source["pairs"].startswith("http"):
                pairs_path = download_path+"/pairs."+source["pairs"].split(".")[-1]
                wget.download(source["pairs"], pairs_path)
                if pairs_path.split(".")[-1] not in ["txt", "csv"]:
                    with gzip.open(pairs_path, 'rb') as f_in:
                        with open(download_path+"/pairs.txt", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(pairs_path)
            else:
                shutil.move(download_path+"/"+source["pairs"], download_path+"/pairs.txt")
        if "pair_process" in source:
            pairs = list()
            with open(download_path+"/pairs.txt", "r") as file:
                for line in file:
                    pair = source["pair_process"](line[:-1].split())
                    if pair is not None:
                        pairs.append(pair)
            os.remove(download_path+"/pairs.txt")
            with open(download_path+"/pairs.txt", "w") as file:
                for pair in pairs:
                    file.write(pair[0]+"\t"+pair[1]+"\n")

        if "node2group" in source:
            groups = dict()
            with open(download_path+"/pairs.txt", "r") as file:
                for line in file:
                    pair = line[:-1].split()
                    for node in pair[0:1]:
                        group = source["node2group"](node)
                        if group is not None:
                            if group not in groups:
                                groups[group] = list()
                            groups[group].append(node)
            with open(download_path+"/groups.txt", "w") as file:
                for group in groups.values():
                    if len(group) > 1:
                        file.write(" ".join(group)+"\n")

        if "features" in source and "groups" not in source:
            features_path = download_path+"/"+source["features"]
            groups = dict()
            features = dict()
            with open(features_path) as features_file:
                for line in features_file:
                    line = line[:-1].split()
                    if "feature_process" in source:
                        line = source["feature_process"](line)
                        if line is None:
                            continue
                    node_id = line[0]
                    group = line[-1]
                    if group not in groups:
                        groups[group] = list()
                    groups[group].append(node_id)
                    features[node_id] = [val.strip() for val in line[1:-1]]
            groups = {group: nodes for group, nodes in groups.items() if len(nodes) > 1}
            with open(download_path+'/groups.txt', 'w', encoding='utf-8') as file:
                for g in groups.values():
                    for uid in g:
                        file.write(str(uid) + '\t')
                    file.write('\n')
            with open(download_path+'/features.txt', 'w', encoding='utf-8') as file:
                for p in features:
                    file.write(str(p) + '\t' + '\t'.join(features[p]) + '\n')

        if "features" in source and "groups" in source:
            if source["features"].startswith("http"):
                pairs_path = download_path+"/features."+source["features"].split(".")[-1]
                wget.download(source["features"], pairs_path)
                if pairs_path.split(".")[-1] not in ["txt", "csv"]:
                    with gzip.open(pairs_path, 'rb') as f_in:
                        with open(download_path+"/features.txt", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(pairs_path)
            else:
                shutil.move(download_path+"/"+source["features"], download_path+"/features.txt")

        if "groups" in source:
            groups_path = download_path+"/groups."+source["groups"].split(".")[-1]
            wget.download(source["groups"], groups_path)
            if groups_path.split(".")[-1] not in ["txt", "csv"]:
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

        if "remove" in source:
            shutil.rmtree(download_path+"/"+source["remove"])
    return credentials
