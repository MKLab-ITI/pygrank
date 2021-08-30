import pygrank as pg

datasets = ["blockmodel", "citeseer", "eucore", "dblp", "amazon"]
pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
algorithms = {
    "ppr0.85": pg.PageRank(alpha=0.85, preprocessor=pre, max_iters=10000, tol=1.E-9),
    "ppr0.9": pg.PageRank(alpha=0.9, preprocessor=pre, max_iters=10000, tol=1.E-9),
    "ppr0.99": pg.PageRank(alpha=0.99, preprocessor=pre, max_iters=10000, tol=1.E-9),
    "hk3": pg.HeatKernel(t=3, preprocessor=pre, max_iters=10000, tol=1.E-9),
    "hk5": pg.HeatKernel(t=5, preprocessor=pre, max_iters=10000, tol=1.E-9),
    "hk7": pg.HeatKernel(t=7, preprocessor=pre, max_iters=10000, tol=1.E-9)
}
tuned = {
    "selected": pg.AlgorithmSelection(algorithms.values(), fraction_of_training=0.8),
    "tuned": pg.ParameterTuner(preprocessor=pre, max_iters=10000, tol=1.E-9, fraction_of_training=0.8),
    "estimated": pg.HopTuner(preprocessor=pre, max_iters=10000, tol=1.E-9, measure=pg.AUC, fraction_of_training=0.8)
}
#algorithms = pg.create_variations(algorithms, {"": pg.Normalize})

loader = pg.load_datasets_all_communities(datasets, min_group_size=50)
pg.benchmark_print(pg.benchmark(algorithms | tuned, loader, pg.AUC, fraction_of_training=.1),
                   decimals=3, delimiter=" & ", end_line="\\\\")