import pygrank as pg
datasets = ["EUCore", "Amazon"]
pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
algs = {"ppr.85": pg.PageRank(.85, preprocessor=pre, tol=1.E-9, max_iters=1000),
        "ppr.99": pg.PageRank(.99, preprocessor=pre, tol=1.E-9, max_iters=1000),
        "hk3": pg.HeatKernel(3, preprocessor=pre, tol=1.E-9, max_iters=1000),
        "hk5": pg.HeatKernel(5, preprocessor=pre, tol=1.E-9, max_iters=1000),
        }

algs = algs | pg.create_variations(algs, {"+Sweep": pg.Sweep})
loader = pg.load_datasets_one_community(datasets)
algs["tuned"] = pg.ParameterTuner(preprocessor=pre, tol=1.E-9, max_iters=1000)
algs["selected"] = pg.AlgorithmSelection(pg.create_demo_filters(preprocessor=pre, tol=1.E-9, max_iters=1000).values())
algs["tuned+Sweep"] = pg.ParameterTuner(ranker_generator=lambda params: pg.Sweep(pg.GenericGraphFilter(params, preprocessor=pre, tol=1.E-9, max_iters=1000)))

for alg in algs.values():
   print(alg.cite())  # prints a list of algorithm citations

pg.benchmark_print(pg.benchmark(algs, loader, pg.AUC, fraction_of_training=.5), delimiter=" & ", end_line="\\\\")
