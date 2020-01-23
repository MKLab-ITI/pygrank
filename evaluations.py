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

def import_SNAP_data(dataset, path='data/', pair_file='pairs.txt', group_file='groups.txt', directed=False, min_group_size=10):
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
                    break
    return G, groups

measure_evaluations = {}
datasets = ["amazon"]
for dataset_name in datasets:
    dataset_name = 'amazon'
    G, groups = import_SNAP_data(dataset_name, min_group_size=3000)
    pre = preprocessor('col', assume_immutability=True)
    pre(G)

    algorithms = {"PPR 0.85": pygrank.algorithms.pagerank.PageRank(alpha=0.85, to_scipy=pre, max_iters=1000),
                  "PPR 0.90": pygrank.algorithms.pagerank.PageRank(alpha=0.9, to_scipy=pre, max_iters=1000),
                  "PPR 0.95": pygrank.algorithms.pagerank.PageRank(alpha=0.95, to_scipy=pre, max_iters=1000),
                  "HK": pygrank.algorithms.pagerank.HeatKernel(to_scipy=pre, max_iters=1000),
                  "PPR+I 0.85": pygrank.algorithms.oversampling.SeedOversampling(pygrank.algorithms.pagerank.PageRank(alpha=0.85, to_scipy=pre, max_iters=1000),method="Neighbors"),
                  "PPR+I 0.90": pygrank.algorithms.oversampling.SeedOversampling(pygrank.algorithms.pagerank.PageRank(alpha=0.9, to_scipy=pre, max_iters=1000),method="Neighbors"),
                  "PPR+I 0.95": pygrank.algorithms.oversampling.SeedOversampling(pygrank.algorithms.pagerank.PageRank(alpha=0.95, to_scipy=pre, max_iters=1000),method="Neighbors"),
                  "PPR+SO 0.85": pygrank.algorithms.oversampling.SeedOversampling(pygrank.algorithms.pagerank.PageRank(alpha=0.85, to_scipy=pre, max_iters=1000)),
                  "PPR+SO 0.90": pygrank.algorithms.oversampling.SeedOversampling(pygrank.algorithms.pagerank.PageRank(alpha=0.9, to_scipy=pre, max_iters=1000)),
                  "PPR+SO 0.95": pygrank.algorithms.oversampling.SeedOversampling(pygrank.algorithms.pagerank.PageRank(alpha=0.95, to_scipy=pre, max_iters=1000)),
                  }
    seeds = [0.001, 0.01, 0.1]
    experiments = list()


    for seed in seeds:
        training_groups, test_groups = pygrank.metrics.utils.split_groups(groups, fraction_of_training=seed)
        test_group_ranks = pygrank.metrics.utils.to_seeds(test_groups)
        measures = {"AUC": pygrank.metrics.multigroup.MultiSupervised(pygrank.metrics.supervised.AUC, test_group_ranks),
                    "NDCG": pygrank.metrics.multigroup.MultiSupervised(pygrank.metrics.supervised.NDCG, test_group_ranks),
                    "Conductance": pygrank.metrics.multigroup.MultiUnsupervised(pygrank.metrics.unsupervised.Conductance, G),
                    "Density": pygrank.metrics.multigroup.MultiUnsupervised(pygrank.metrics.unsupervised.Density, G),
                    "LinkAUC": pygrank.metrics.multigroup.LinkAUC(G)}
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

for name, eval in measure_evaluations.items():
    print(name, eval)

print("AUC vs LinkAUC", scipy.stats.spearmanr(measure_evaluations["AUC"], measure_evaluations["LinkAUC"]))
print("AUC vs Conductance", scipy.stats.spearmanr(measure_evaluations["AUC"], measure_evaluations["Conductance"]))