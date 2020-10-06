"""
This file covers the experiments of the paper: Unsupervised evaluation of multiple node ranks by reconstructing local structures
"""

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


def import_SNAP_data(dataset, path='data/', pair_file='pairs.txt', group_file='groups.txt', directed=False, min_group_size=10, max_group_number=10, import_label_file=False):
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
        pass

    else:
        with open(path+dataset+'/'+group_file, 'r', encoding='utf-8') as file:
            for line in file:
                if line[0] != '#':
                    group = [item for item in line[:-1].split('\t') if len(item) > 0 and item in G]
                    if len(group) >= min_group_size:
                        groups[len(groups)] = group
                        if len(groups) >= max_group_number:
                            break
    return G, groups

if __name__ == "__main__":
    measure_evaluations = {}
    datasets = ['amazon']
    max_iters = 10000
    for dataset_name in datasets:
        G, groups = import_SNAP_data(dataset_name, min_group_size=5000)#12000 for dblp, 5000 for amazon
        group_sets = [set(group) for group in groups.values()]
        for group in group_sets:
            print(len(group))
        count = sum(1 for u, v in G.edges() if sum(1 for group in group_sets if u in group and v in group) > 0)
        print('Homophily', count / float(G.number_of_edges()))
        seeds = [0.001, 0.01, 0.1, 0.25, 0.5]
        print('Number of groups', len(groups))
        for seed in seeds:
            pre = preprocessor('col', assume_immutability=True)
            preL = preprocessor('symmetric', assume_immutability=True)
            pre(G)
            tol = 1.E-6
            base_algorithms = {"PPRL 0.85": pygrank.algorithms.pagerank.PageRank(alpha=0.85, to_scipy=preL, max_iters=max_iters, tol=tol),
                          "PPRL 0.90": pygrank.algorithms.pagerank.PageRank(alpha=0.9, to_scipy=preL, max_iters=max_iters, tol=tol),
                          "PPRL 0.95": pygrank.algorithms.pagerank.PageRank(alpha=0.95, to_scipy=preL, max_iters=max_iters, tol=tol),
                          "PPRL 0.99": pygrank.algorithms.pagerank.PageRank(alpha=0.99, to_scipy=preL, max_iters=max_iters, tol=tol),
                           "PPR 0.85": pygrank.algorithms.pagerank.PageRank(alpha=0.85, to_scipy=pre, max_iters=max_iters, tol=tol),
                           "PPR 0.90": pygrank.algorithms.pagerank.PageRank(alpha=0.9, to_scipy=pre, max_iters=max_iters, tol=tol),
                           "PPR 0.95": pygrank.algorithms.pagerank.PageRank(alpha=0.95, to_scipy=pre, max_iters=max_iters, tol=tol),
                           "PPR 0.99": pygrank.algorithms.pagerank.PageRank(alpha=0.99, to_scipy=pre, max_iters=max_iters, tol=tol),
                           "AbsorbL 0.85": pygrank.algorithms.pagerank.AbsorbingRank(alpha=0.85, to_scipy=preL, max_iters=max_iters, tol=tol),
                           "AbsorbL 0.90": pygrank.algorithms.pagerank.AbsorbingRank(alpha=0.9, to_scipy=preL, max_iters=max_iters, tol=tol),
                           "AbsorbL 0.95": pygrank.algorithms.pagerank.AbsorbingRank(alpha=0.95, to_scipy=preL, max_iters=max_iters, tol=tol),
                           "AbsorbL 0.99": pygrank.algorithms.pagerank.AbsorbingRank(alpha=0.99, to_scipy=preL, max_iters=max_iters, tol=tol),
                           "Absorb 0.85": pygrank.algorithms.pagerank.AbsorbingRank(alpha=0.85, to_scipy=pre, max_iters=max_iters, tol=tol),
                           "Absorb 0.90": pygrank.algorithms.pagerank.AbsorbingRank(alpha=0.9, to_scipy=pre, max_iters=max_iters, tol=tol),
                           "Absorb 0.95": pygrank.algorithms.pagerank.AbsorbingRank(alpha=0.95, to_scipy=pre, max_iters=max_iters, tol=tol),
                           "Absorb 0.99": pygrank.algorithms.pagerank.AbsorbingRank(alpha=0.99, to_scipy=pre, max_iters=max_iters, tol=tol),
                          "HK1": pygrank.algorithms.pagerank.HeatKernel(t=1, to_scipy=pre, max_iters=max_iters, tol=tol),
                          "HK3": pygrank.algorithms.pagerank.HeatKernel(t=3, to_scipy=pre, max_iters=max_iters, tol=tol),
                          "HK5": pygrank.algorithms.pagerank.HeatKernel(t=5, to_scipy=pre, max_iters=max_iters, tol=tol),
                          "HK7": pygrank.algorithms.pagerank.HeatKernel(t=7, to_scipy=pre, max_iters=max_iters, tol=tol),
                          "HKL1": pygrank.algorithms.pagerank.HeatKernel(t=1, to_scipy=preL, max_iters=max_iters, tol=tol),
                          "HKL3": pygrank.algorithms.pagerank.HeatKernel(t=3, to_scipy=preL, max_iters=max_iters, tol=tol),
                          "HKL5": pygrank.algorithms.pagerank.HeatKernel(t=5, to_scipy=preL, max_iters=max_iters, tol=tol),
                          "HKL7": pygrank.algorithms.pagerank.HeatKernel(t=7, to_scipy=preL, max_iters=max_iters, tol=tol),
                           "HPRL 0.85": pygrank.algorithms.pagerank.BiasedKernel(alpha=0.85, to_scipy=preL, max_iters=max_iters, tol=tol),
                           "HPRL 0.90": pygrank.algorithms.pagerank.BiasedKernel(alpha=0.9, to_scipy=preL, max_iters=max_iters, tol=tol),
                           "HPRL 0.95": pygrank.algorithms.pagerank.BiasedKernel(alpha=0.95, to_scipy=preL, max_iters=max_iters, tol=tol),
                           "HPRL 0.99": pygrank.algorithms.pagerank.BiasedKernel(alpha=0.99, to_scipy=preL, max_iters=max_iters, tol=tol),
                           "HPR 0.85": pygrank.algorithms.pagerank.BiasedKernel(alpha=0.85, to_scipy=pre, max_iters=max_iters, tol=tol),
                           "HPR 0.90": pygrank.algorithms.pagerank.BiasedKernel(alpha=0.9, to_scipy=pre, max_iters=max_iters, tol=tol),
                           "HPR 0.95": pygrank.algorithms.pagerank.BiasedKernel(alpha=0.95, to_scipy=pre, max_iters=max_iters, tol=tol),
                           "HPR 0.99": pygrank.algorithms.pagerank.BiasedKernel(alpha=0.99, to_scipy=pre, max_iters=max_iters, tol=tol),
                        }
            algorithms = dict()
            for alg_name, alg in base_algorithms.items():
                algorithms[alg_name] = alg
                algorithms[alg_name+" SO"] = pygrank.algorithms.oversampling.SeedOversampling(alg, method="safe")
                algorithms[alg_name+" I"] = pygrank.algorithms.oversampling.SeedOversampling(alg, method="neighbors")
                #algorithms[alg_name+" T"] = pygrank.algorithms.oversampling.SeedOversampling(alg, method="top")
            experiments = list()

            max_positive_samples = 2000
            training_groups, test_groups = pygrank.metrics.utils.split_groups(groups, fraction_of_training=seed)
            test_group_ranks = pygrank.metrics.utils.to_seeds(test_groups)
            measures = {"AUC": pygrank.metrics.multigroup.MultiSupervised(pygrank.metrics.supervised.AUC, test_group_ranks),
                        "NDCG": pygrank.metrics.multigroup.MultiSupervised(pygrank.metrics.supervised.NDCG, test_group_ranks),
                        "Conductance": pygrank.metrics.multigroup.MultiUnsupervised(pygrank.metrics.unsupervised.Conductance, G),
                        "ClusteringCoefficient": pygrank.metrics.multigroup.ClusteringCoefficient(G, similarity="cos", max_positive_samples=max_positive_samples),
                        "Density": pygrank.metrics.multigroup.MultiUnsupervised(pygrank.metrics.unsupervised.Density, G),
                        "Modularity": pygrank.metrics.multigroup.MultiUnsupervised(pygrank.metrics.unsupervised.Modularity, G, max_positive_samples=max_positive_samples),
                        #"DotLinkAUC": pygrank.metrics.multigroup.LinkAUC(G, similarity="dot", max_positive_samples=max_positive_samples, max_negative_samples=max_positive_samples),
                        "CosLinkAUC": pygrank.metrics.multigroup.LinkAUC(G, similarity="cos", max_positive_samples=max_positive_samples, max_negative_samples=max_positive_samples, seed=1),
                        "HopAUC": pygrank.metrics.multigroup.LinkAUC(G, similarity="cos", hops=2,max_positive_samples=max_positive_samples, max_negative_samples=max_positive_samples, seed=1),
                        "LinkCE": pygrank.metrics.multigroup.LinkAUC(G, evaluation="CrossEntropy", similarity="cos", hops=1,max_positive_samples=max_positive_samples, max_negative_samples=max_positive_samples, seed=1),
                        "HopCE": pygrank.metrics.multigroup.LinkAUC(G, evaluation="CrossEntropy", similarity="cos", hops=2,max_positive_samples=max_positive_samples, max_negative_samples=max_positive_samples, seed=1)
                        }
            if len(measure_evaluations) == 0:
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
                #print(experiment_outcome)
            print("-----")
            for measure in measure_evaluations:
                if measure != 'NDCG' and measure != 'AUC':
                    print("NDCG vs", measure, scipy.stats.spearmanr(measure_evaluations["NDCG"], measure_evaluations[measure]))
            print('-----')
            for measure in measure_evaluations:
                if measure != 'NDCG' and measure != 'AUC':
                    print("AUC vs", measure, scipy.stats.spearmanr(measure_evaluations["AUC"], measure_evaluations[measure]))
            print('This is the latest version')

            print('-----')
            for name, eval in measure_evaluations.items():
                print(name, '=', eval, ';')


            #measure_evaluations = dict()