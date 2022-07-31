import pygrank as pg


datasets = ["friendster"]
pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")  # common preprocessor
algs = {"ppr.85": pg.PageRank(.85, preprocessor=pre),
        "ppr.99": pg.PageRank(.99, preprocessor=pre, max_iters=1000),
        "hk3": pg.HeatKernel(3, preprocessor=pre),
        "hk5": pg.HeatKernel(5, preprocessor=pre),
        "tuned": pg.ParameterTuner(preprocessor=pre)}
loader = pg.load_datasets_one_community(datasets)
pg.benchmark_print(pg.benchmark(algs, loader, pg.AUC, fraction_of_training=.5))
