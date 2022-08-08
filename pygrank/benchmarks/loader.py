from pygrank.fastgraph import fastgraph as nx
import os
import numpy as np
from pygrank.core import to_signal, utils
from pygrank.algorithms import call
from pygrank.benchmarks.download import download_dataset
from typing import Iterable, Union


def import_snap_format_dataset(dataset: str,
                               path: str = 'data',
                               pair_file: str = 'pairs.txt',
                               group_file: str = 'groups.txt',
                               directed: bool = False,
                               min_group_size: float = 0.01,
                               max_group_number: int = 20,
                               graph_api=nx,
                               verbose=True):
    if not os.path.isdir(path):   # pragma: no cover
        path = "../"+path
    download_dataset(dataset, path=path)
    if verbose:
        utils.log(f"Loading {dataset} graph")
    G = graph_api.Graph(False) if directed else graph_api.Graph()
    groups = {}
    with open(path+'/'+dataset+'/'+pair_file, 'r', encoding='utf-8') as file:
        for line in file:
            if len(line) != 0 and line[0] != '#':
                splt = line[:-1].split()
                if len(splt) > 1:
                    G.add_edge(splt[0], splt[1])
    if min_group_size < 1:
        min_group_size *= len(G)
    if verbose:
        utils.log(f"Loading {dataset} communities")
    if group_file is not None and os.path.isfile(path+'/'+dataset+'/'+group_file):
        with open(path+'/'+dataset+'/'+group_file, 'r', encoding='utf-8') as file:
            for line in file:
                if line[0] != '#':
                    group = [item for item in line[:-1].split() if len(item) > 0 and item in G]
                    if len(group) >= min_group_size:
                        groups[len(groups)] = group
                        if verbose:
                            utils.log(f"Loaded {dataset} communities {len(groups)}/{max_group_number}")
                        if len(groups) >= max_group_number:
                            break
    if verbose:
        utils.log()
    return G, groups


def _import_features(dataset: str,
                     path: str = 'data',
                     feature_file: str = 'features.txt'):
    if not os.path.isdir(path):   # pragma: no cover
        path = "../"+path
    features = dict()
    pos_dict = dict()
    feature_length = 0
    with open(path+'/'+dataset+'/'+feature_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line[:-1].split()
            if "=" in line[1]:   # pragma: no cover
                found = dict()
                for feat in line[1:]:
                    feat = feat.split("=")
                    if feat[0] not in pos_dict:
                        pos_dict[feat[0]] = len(pos_dict)
                    found[pos_dict[feat[0]]] = float(feat[1])
                features[line[0]] = [found.get(i, 0.) for i in range(max(found.keys()))]
            else:
                features[line[0]] = [float(val) for val in line[1:]]
            feature_length = max(feature_length, len(features[line[0]]))
    features = {v: row+[0]*(feature_length-len(row)) for v, row in features.items()}
    return features


def _preprocess_features(features: np.ndarray):
    """Row-normalizes a feature matrix.
    Follows the implementation of: https://github.com/tkipf/gcn/blob/master/gcn/utils.py

    Args:
        features: A numpy matrix whose rows correspond to node features.
    Returns:
        The normalized feature matrix.
    """


    #r_inv = np.asarray(np.sum(features, axis=0), np.float64)
    #r_inv[r_inv != 0] = np.power(r_inv[r_inv != 0], -1)
    #features = features * r_inv
    #return features
    import scipy.sparse
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = scipy.sparse.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features




def load_feature_dataset(dataset: str,
                         path: str = 'data',
                         groups_no_labels = False,
                         **kwargs):
    """
    Imports a dataset comprising node features. Features and labels are organized as numpy matrix.
    This tries to automatically download the dataset first if not found.

    Args:
        dataset: The dataset'personalization name. Corresponds to a folder name in which the dataset is stored.
        path: The dataset'personalization path in which *dataset* is a folder. If path not found in the file system,
            "../" is prepended. Default is "data".
        kwargs: Optional. Additional arguments to pass to *import_snap_format_dataset*.
    Returns:
        graph: A graph of node relations. Nodes are indexed in the order the graph is traversed.
        features: A column-normalized numpy matrix whose rows correspond to node features.
        labels: A numpy matrix whose rows correspond to one-hot encodings of node labels.
    """
    graph, groups = call(import_snap_format_dataset, kwargs, [dataset, path])
    features = call(_import_features, kwargs, [dataset, path])
    feature_dims = len(features[list(features.keys())[0]])
    features = np.array([features.get(v, [0] * feature_dims) for v in graph], dtype=np.float64)
    features = _preprocess_features(features)
    labels = groups if groups_no_labels else np.array([to_signal(graph, group).np for group in groups.values()], dtype=np.float64).transpose()
    return graph, features, labels


def load_datasets_multiple_communities(datasets: Iterable[str], **kwargs):
    for dataset in datasets:
        graph, groups = import_snap_format_dataset(dataset, **kwargs)
        if len(groups) != 0:
            yield dataset, graph, groups


def load_datasets_all_communities(datasets: Iterable[str], **kwargs):
    for dataset, graph, groups in load_datasets_multiple_communities(datasets, **kwargs):
        for group_id, group in groups.items():
            yield dataset+str(group_id), graph, group


def load_datasets_graph(datasets: Iterable[str], **kwargs):
    """
    Iterates through all available datasets that exhibit structural communities and loads them with
    *import_snap_format_dataset* for experiments.
    Found datasets are yielded to iterate through.

    Args:
        datasets: A iterable of dataset names corresponding to a folder name in which the dataset is stored.
        kwargs: Additional keyword arguments to pass to the method `import_snap_format_dataset`.
    Yields:
        graph: A graph of node relations. Nodes are indexed in the order the graph is traversed.

    Example:
        >>> import pygrank as pg
        >>> for graph, group in pg.load_datasets_one_community(pg.downloadable_datasets()):
        >>>     ...
    """
    datasets = [(dataset, 0) if len(dataset) != 2 else dataset for dataset in datasets]
    for dataset, group_id in datasets:
        graph, _ = import_snap_format_dataset(dataset,
                                              max_group_number=0,
                                              **kwargs)
        yield graph


def load_datasets_one_community(datasets: Iterable[str], **kwargs):
    """
    Iterates through all available datasets that exhibit structural communities and loads them with
    *import_snap_format_dataset* for experiments.
    Found datasets are yielded to iterate through.

    Args:
        datasets: A iterable of dataset names corresponding to a folder name in which the dataset is stored.
        kwargs: Additional keyword arguments to pass to the method `import_snap_format_dataset`.
    Yields:
        graph: A graph of node relations. Nodes are indexed in the order the graph is traversed.
        group: The first structural community found in the dataset.

    Example:
        >>> import pygrank as pg
        >>> for graph, group in pg.load_datasets_one_community(pg.downloadable_datasets()):
        >>>     ...
    """
    datasets = [(dataset, 0) if len(dataset) != 2 else dataset for dataset in datasets]
    last_loaded_dataset = None
    for dataset, group_id in datasets:
        if last_loaded_dataset != dataset:
            max_group_number = 1 + max(group_id for dat, group_id in datasets if dat == dataset)
            graph, groups = import_snap_format_dataset(dataset,
                                                       max_group_number=max_group_number,
                                                       **kwargs)
            last_loaded_dataset = dataset
        if len(groups) > group_id:
            group = set(groups[group_id])
            yield dataset, graph, group
