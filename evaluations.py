import networkx as nx
from pygrank.algorithms.utils import preprocessor
import pygrank.algorithms.pagerank
import pygrank.algorithms.oversampling
import pygrank.algorithms.postprocess
import pygrank.metrics.utils
import pygrank.metrics.unsupervised
import pygrank.metrics.supervised
import pygrank.metrics.multigroup
import scipy.stats


def import_SNAP_data(dataset, path='data/', pair_file='pairs.txt', group_file='groups.txt', directed=False, min_group_size=10, max_group_number=10):
    G = nx.DiGraph() if directed else nx.Graph()
    groups = {}
    with open(path+dataset+'/'+pair_file, 'r', encoding='utf-8') as file:
        for line in file:
            if len(line) != 0 and line[0] != '#':
                splt = line[:-1].split('\t')
                if len(splt) == 0:
                    continue
                G.add_edge(splt[0], splt[1])
    with open(path+dataset+'/'+group_file, 'r', encoding='utf-8') as file:
        for line in file:
            if line[0] != '#':
                group = [item for item in line[:-1].split('\t') if len(item) > 0 and item in G]
                if len(group) >= min_group_size:
                    groups[len(groups)] = group
                    if len(groups) >= max_group_number:
                        break
    return G, groups

measure_evaluations = {}
datasets = ['amazon']
for dataset_name in datasets:
    G, groups = import_SNAP_data(dataset_name, min_group_size=1000)
    seeds = [0.001, 0.01, 0.1, 0.25, 0.5]
    print('Number of groups', len(groups))
    for seed in seeds:
        pre = preprocessor('col', assume_immutability=True)
        preL = preprocessor('symmetric', assume_immutability=True)
        pre(G)

        base_algorithms = {"PPRL 0.85": pygrank.algorithms.pagerank.PageRank(alpha=0.85, to_scipy=preL, max_iters=1000),
                      "PPRL 0.90": pygrank.algorithms.pagerank.PageRank(alpha=0.9, to_scipy=preL, max_iters=1000),
                      "PPRL 0.95": pygrank.algorithms.pagerank.PageRank(alpha=0.95, to_scipy=preL, max_iters=1000),
                      "PPRL 0.99": pygrank.algorithms.pagerank.PageRank(alpha=0.99, to_scipy=preL, max_iters=1000),
                       "PPR 0.85": pygrank.algorithms.pagerank.PageRank(alpha=0.85, to_scipy=pre, max_iters=1000),
                       "PPR 0.90": pygrank.algorithms.pagerank.PageRank(alpha=0.9, to_scipy=pre, max_iters=1000),
                       "PPR 0.95": pygrank.algorithms.pagerank.PageRank(alpha=0.95, to_scipy=pre, max_iters=1000),
                       "PPR 0.99": pygrank.algorithms.pagerank.PageRank(alpha=0.99, to_scipy=pre, max_iters=1000),
                       "HPRL 0.85": pygrank.algorithms.pagerank.BiasedKernel(alpha=0.85, to_scipy=preL, max_iters=1000),
                       "HPRL 0.90": pygrank.algorithms.pagerank.BiasedKernel(alpha=0.9, to_scipy=preL, max_iters=1000),
                       "HPRL 0.95": pygrank.algorithms.pagerank.BiasedKernel(alpha=0.95, to_scipy=preL, max_iters=1000),
                       "HPRL 0.99": pygrank.algorithms.pagerank.BiasedKernel(alpha=0.99, to_scipy=preL, max_iters=1000),
                       "HPR 0.85": pygrank.algorithms.pagerank.BiasedKernel(alpha=0.85, to_scipy=pre, max_iters=1000),
                       "HPR 0.90": pygrank.algorithms.pagerank.BiasedKernel(alpha=0.9, to_scipy=pre, max_iters=1000),
                       "HPR 0.95": pygrank.algorithms.pagerank.BiasedKernel(alpha=0.95, to_scipy=pre, max_iters=1000),
                       "HPR 0.99": pygrank.algorithms.pagerank.BiasedKernel(alpha=0.99, to_scipy=pre, max_iters=1000),
                      "HK1": pygrank.algorithms.pagerank.HeatKernel(t=1, to_scipy=pre, max_iters=1000),
                      "HK3": pygrank.algorithms.pagerank.HeatKernel(t=3, to_scipy=pre, max_iters=1000),
                      "HK5": pygrank.algorithms.pagerank.HeatKernel(t=5, to_scipy=pre, max_iters=1000),
                      "HK7": pygrank.algorithms.pagerank.HeatKernel(t=7, to_scipy=pre, max_iters=1000),
                      "HKL1": pygrank.algorithms.pagerank.HeatKernel(t=1, to_scipy=preL, max_iters=1000),
                      "HKL3": pygrank.algorithms.pagerank.HeatKernel(t=3, to_scipy=preL, max_iters=1000),
                      "HKL5": pygrank.algorithms.pagerank.HeatKernel(t=5, to_scipy=preL, max_iters=1000),
                      "HKL7": pygrank.algorithms.pagerank.HeatKernel(t=7, to_scipy=preL, max_iters=1000)}
        algorithms = dict()
        for alg_name, alg in base_algorithms.items():
            algorithms[alg_name] = alg
            algorithms[alg_name+" SO"] = pygrank.algorithms.oversampling.SeedOversampling(alg, method="safe")
            algorithms[alg_name+" I"] = pygrank.algorithms.oversampling.SeedOversampling(alg, method="neighbors")
        experiments = list()

        training_groups, test_groups = pygrank.metrics.utils.split_groups(groups, fraction_of_training=seed)
        test_group_ranks = pygrank.metrics.utils.to_seeds(test_groups)
        measures = {"AUC": pygrank.metrics.multigroup.MultiSupervised(pygrank.metrics.supervised.AUC, test_group_ranks),
                    "NDCG": pygrank.metrics.multigroup.MultiSupervised(pygrank.metrics.supervised.NDCG, test_group_ranks),
                    "Conductance": pygrank.metrics.multigroup.MultiUnsupervised(pygrank.metrics.unsupervised.Conductance, G),
                    "ClusteringCoefficient": pygrank.metrics.multigroup.ClusteringCoefficient(G, similarity="cos"),
                    "Density": pygrank.metrics.multigroup.MultiUnsupervised(pygrank.metrics.unsupervised.Density, G),
                    "Modularity": pygrank.metrics.multigroup.MultiUnsupervised(pygrank.metrics.unsupervised.Modularity, G),
                    "DotLinkAUC": pygrank.metrics.multigroup.LinkAUC(G, similarity="dot"),
                    "CosLinkAUC": pygrank.metrics.multigroup.LinkAUC(G, similarity="cos"),
                    "HopAUC": pygrank.metrics.multigroup.LinkAUC(G, similarity="cos", hops=2)}
        if len(measure_evaluations)==0:
            for measure_name in measures.keys():
                measure_evaluations[measure_name] = list()
        for alg_name, alg in algorithms.items():
            alg = pygrank.algorithms.postprocess.Normalize(alg)
            experiment_outcome = dataset_name+" & "+str(seed)+" & "+alg_name;
            ranks = {group_id: alg.rank(G, {v: 1 for v in group}) for group_id, group in training_groups.items()}
            print(experiment_outcome)
            for measure_name, measure in measures.items():
                measure_outcome = measure.evaluate(ranks)
                measure_evaluations[measure_name].append(measure_outcome)
                print("\t", measure_name, measure_outcome)
                experiment_outcome += " & "+str(measure_outcome)
            print(experiment_outcome)
            print("-----")

            print("AUC vs CosLinkAUC", scipy.stats.spearmanr(measure_evaluations["AUC"], measure_evaluations["CosLinkAUC"]))
            print("AUC vs DotLinkAUC", scipy.stats.spearmanr(measure_evaluations["AUC"], measure_evaluations["DotLinkAUC"]))
            print("AUC vs HopAUC", scipy.stats.spearmanr(measure_evaluations["AUC"], measure_evaluations["HopAUC"]))
            print("AUC vs ClusteringCoefficient", scipy.stats.spearmanr(measure_evaluations["AUC"], measure_evaluations["ClusteringCoefficient"]))
            print("AUC vs Conductance", scipy.stats.spearmanr(measure_evaluations["AUC"], measure_evaluations["Conductance"]))
            print("AUC vs Density", scipy.stats.spearmanr(measure_evaluations["AUC"], measure_evaluations["Density"]))
            print("AUC vs Modularity", scipy.stats.spearmanr(measure_evaluations["AUC"], measure_evaluations["Modularity"]))
            print('-----')
            print("NDCG vs CosLinkAUC", scipy.stats.spearmanr(measure_evaluations["NDCG"], measure_evaluations["CosLinkAUC"]))
            print("NDCG vs DotLinkAUC", scipy.stats.spearmanr(measure_evaluations["NDCG"], measure_evaluations["DotLinkAUC"]))
            print("NDCG vs HopAUC", scipy.stats.spearmanr(measure_evaluations["NDCG"], measure_evaluations["HopAUC"]))
            print("NDCG vs ClusteringCoefficient", scipy.stats.spearmanr(measure_evaluations["NDCG"], measure_evaluations["ClusteringCoefficient"]))
            print("NDCG vs Conductance", scipy.stats.spearmanr(measure_evaluations["NDCG"], measure_evaluations["Conductance"]))
            print("NDCG vs Density", scipy.stats.spearmanr(measure_evaluations["NDCG"], measure_evaluations["Density"]))
            print("NDCG vs Modularity", scipy.stats.spearmanr(measure_evaluations["NDCG"], measure_evaluations["Modularity"]))
            print('-----')
            print("CosLinkAUC vs Density", scipy.stats.spearmanr(measure_evaluations["CosLinkAUC"], measure_evaluations["Density"]))
            print("DotLinkAUC vs Density", scipy.stats.spearmanr(measure_evaluations["DotLinkAUC"], measure_evaluations["Density"]))


print('-----')
for name, eval in measure_evaluations.items():
    print(name, eval)