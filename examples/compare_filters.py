import pygrank as pg

datasets = ["bigraph", "citeseer", "amazon", "dblp"]
pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
algorithms = {
    "ppr0.85": pg.PageRank(alpha=0.85, preprocessor=pre, max_iters=10000, tol=1.E-9),
    "ppr0.99": pg.PageRank(alpha=0.99, preprocessor=pre, max_iters=10000, tol=1.E-9),
    "hk3": pg.HeatKernel(t=3, preprocessor=pre, max_iters=10000, tol=1.E-9),
    "hk5": pg.HeatKernel(t=5, preprocessor=pre, max_iters=10000, tol=1.E-9),
    "tuned": pg.ParameterTuner(preprocessor=pre, max_iters=10000, tol=1.E-9),
    "hoptuned": pg.HopTuner(preprocessor=pre, max_iters=10000, tol=1.E-9),
}
algorithms = pg.create_variations(algorithms, {"": pg.Normalize})

loader = pg.load_datasets_all_communities(datasets)
pg.benchmark_print(pg.benchmark(algorithms, loader, pg.AUC, fraction_of_training=.8))