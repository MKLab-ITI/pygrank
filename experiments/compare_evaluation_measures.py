"""
This file covers the experiments of the paper: Unsupervised evaluation of multiple node ranks by reconstructing local structures
"""

from pygrank.algorithms import preprocessor
import pygrank.algorithms.postprocess
import pygrank.measures.utils
import pygrank.measures.multigroup
import scipy.stats
from experiments.importer import import_SNAP

if __name__ == "__main__":
    measure_evaluations = {}
    datasets = ['amazon']
    max_iters = 10000
    for dataset_name in datasets:
        G, groups = import_SNAP(dataset_name, min_group_size=5000)#12000 for dblp, 5000 for amazon
        group_sets = [set(group) for group in groups.values()]
        for group in group_sets:
            print(len(group))
        count = sum(1 for u, v in G.edges() if sum(1 for group in group_sets if u in group and v in group) > 0)
        print('Homophily', count / float(G.number_of_edges()))
        seeds = [0.001, 0.01, 0.1, 0.25, 0.5]
        print('Number of groups', len(groups))
        for seed in seeds:
            create_variations(create_many_filters(G), seed_oversampling_variations())
            algorithms = dict()
            for alg_name, alg in base_algorithms.items():
                algorithms[alg_name] = alg
                algorithms[alg_name+" SO"] = pygrank.algorithms.oversampling.SeedOversampling(alg, method="safe")
                algorithms[alg_name+" I"] = pygrank.algorithms.oversampling.SeedOversampling(alg, method="neighbors")
                #algorithms[alg_name+" T"] = pygrank.algorithms.oversampling.SeedOversampling(alg, method="top")
            experiments = list()

            max_positive_samples = 2000
            training_groups, test_groups = pygrank.measures.utils.split(groups, training_samples=seed)
            test_group_ranks = pygrank.measures.utils.to_seeds(test_groups)
            measures = {"AUC": pygrank.measures.multigroup.MultiSupervised(pygrank.measures.supervised.AUC, test_group_ranks),
                        "NDCG": pygrank.measures.multigroup.MultiSupervised(pygrank.measures.supervised.NDCG, test_group_ranks),
                        "Conductance": pygrank.measures.multigroup.MultiUnsupervised(pygrank.measures.unsupervised.Conductance, G),
                        "ClusteringCoefficient": pygrank.measures.multigroup.ClusteringCoefficient(G, similarity="cos", max_positive_samples=max_positive_samples),
                        "Density": pygrank.measures.multigroup.MultiUnsupervised(pygrank.measures.unsupervised.Density, G),
                        "Modularity": pygrank.measures.multigroup.MultiUnsupervised(pygrank.measures.unsupervised.Modularity, G, max_positive_samples=max_positive_samples),
                        #"DotLinkAUC": pygrank.metrics.multigroup.LinkAUC(G, similarity="dot", max_positive_samples=max_positive_samples, max_negative_samples=max_positive_samples),
                        "CosLinkAUC": pygrank.measures.multigroup.LinkAssessment(G, similarity="cos", max_positive_samples=max_positive_samples, max_negative_samples=max_positive_samples, seed=1),
                        "HopAUC": pygrank.measures.multigroup.LinkAssessment(G, similarity="cos", hops=2, max_positive_samples=max_positive_samples, max_negative_samples=max_positive_samples, seed=1),
                        "LinkCE": pygrank.measures.multigroup.LinkAssessment(G, measure="CrossEntropy", similarity="cos", hops=1, max_positive_samples=max_positive_samples, max_negative_samples=max_positive_samples, seed=1),
                        "HopCE": pygrank.measures.multigroup.LinkAssessment(G, measure="CrossEntropy", similarity="cos", hops=2, max_positive_samples=max_positive_samples, max_negative_samples=max_positive_samples, seed=1)
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