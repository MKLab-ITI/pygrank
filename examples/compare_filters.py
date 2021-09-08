import pygrank as pg


datasets = ["amazon", "blockmodel", "citeseer", "dblp", "eucore", "maven"]
#datasets = ["citeseer", "eucore", "dblp"]
pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
tol = 1.E-9
algorithms = {
    "ppr0.85": pg.PageRank(alpha=0.85, preprocessor=pre, max_iters=1000, tol=tol),
    "ppr0.9": pg.PageRank(alpha=0.9, preprocessor=pre, max_iters=1000, tol=tol),
    "ppr0.99": pg.PageRank(alpha=0.99, preprocessor=pre, max_iters=1000, tol=tol),
    "hk3": pg.HeatKernel(t=3, preprocessor=pre, max_iters=1000, tol=tol),
    "hk5": pg.HeatKernel(t=5, preprocessor=pre, max_iters=1000, tol=tol),
    "hk7": pg.HeatKernel(t=7, preprocessor=pre, max_iters=1000, tol=tol)
}
tuned = {
    "selected": pg.AlgorithmSelection(algorithms.values(), fraction_of_training=0.8),
    "tuned": pg.ParameterTuner(preprocessor=pre, fraction_of_training=0.8,
                               max_vals=[1]*20, min_vals=[0]*20,
                               #krylov_dims=10,

                               deviation_tol=0.005,
                               error_type="iters", max_iters=20),
    "arnoldi": pg.HopTuner(preprocessor=pre, error_type="iters", max_iters=20,
                             measure=pg.Cos, tunable_offset=pg.AUC,
                             autoregression=0,
                             num_parameters=20,
                             basis="arnoldi"
                             ),
    #"estimated": pg.HopTuner(preprocessor=pre, error_type="iters", max_iters=20,
    #                         measure=pg.AUC, tunable_offset=pg.AUC,
    #                         autoregression=0,
    #                         num_parameters=20
    #                         )
}

loader = pg.load_datasets_all_communities(datasets, min_group_size=50)
pg.benchmark_print(pg.benchmark(algorithms | tuned, loader, pg.AUC, fraction_of_training=.8, seed=list(range(5))),
                   decimals=3, delimiter=" & ", end_line="\\\\")

