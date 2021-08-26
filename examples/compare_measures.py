import pygrank as pg

loader = list(pg.load_datasets_multiple_communities(["graph9", "bigraph", "citeseer"]))
algorithms = pg.create_variations(pg.create_many_filters(), pg.create_many_variation_types())
algorithms = pg.create_variations(algorithms, {"": pg.Normalize}) # add normalization to all algorithms

measures = {"AUC": lambda ground_truth, exlude: pg.MultiSupervised(pg.AUC, ground_truth, exlude),
            "NDCG": lambda ground_truth, exlude: pg.MultiSupervised(pg.NDCG, ground_truth, exlude),
            "Density": lambda graph: pg.MultiUnsupervised(pg.Density, graph),
            "Modularity": lambda graph: pg.MultiUnsupervised(pg.Modularity, graph),
            "CCcos": lambda graph: pg.ClusteringCoefficient(graph, similarity="cos"),
            "CCdot": lambda graph: pg.ClusteringCoefficient(graph, similarity="dot"),
            "LinkAUCcos": lambda graph: pg.LinkAssessment(graph, similarity="cos"),
            "LinkAUCdot": lambda graph: pg.LinkAssessment(graph, similarity="dot"),
            "HopAUCcos": lambda graph: pg.LinkAssessment(graph, similarity="cos", hops=2),
            "HopAUCdot": lambda graph: pg.LinkAssessment(graph, similarity="dot", hops=2),
            }

scores = {measure: pg.benchmark_scores(pg.benchmark(algorithms, loader, measures[measure])) for measure in measures}
evaluations_vs_auc = dict()
evaluations_vs_ndcg = dict()
for measure in measures:
    evaluations_vs_auc[measure] = abs(pg.SpearmanCorrelation(scores["AUC"])(scores[measure]))
    evaluations_vs_ndcg[measure] = abs(pg.SpearmanCorrelation(scores["NDCG"])(scores[measure]))

pg.benchmark_print([("Measure", "AUC corr", "NDCG corr")]+[(measure, evaluations_vs_auc[measure], evaluations_vs_ndcg[measure])for measure in measures])