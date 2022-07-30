import pygrank as pg

datasets = ["amazon", "citeseer", "maven"]
community_size = 500

pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
convergence = {"tol": 1.E-9, "max_iters": 10000}
#convergence = {"error_type": "iters", "max_iters": 41}

algorithms = {
    "ppr0.5": pg.PageRank(alpha=0.5, preprocessor=pre, **convergence),
    "ppr0.85": pg.PageRank(alpha=0.85, preprocessor=pre, **convergence),
    "ppr0.9": pg.PageRank(alpha=0.9, preprocessor=pre, **convergence),
    "ppr0.99": pg.PageRank(alpha=0.99, preprocessor=pre, **convergence),
    "hk2": pg.HeatKernel(t=2, preprocessor=pre, **convergence),
    "hk3": pg.HeatKernel(t=3, preprocessor=pre, **convergence),
    "hk5": pg.HeatKernel(t=5, preprocessor=pre, **convergence),
    "hk7": pg.HeatKernel(t=7, preprocessor=pre, **convergence),
}

postprocessor = pg.Tautology
algorithms = pg.benchmarks.create_variations(algorithms, postprocessor)
measure = pg.AUC
optimization = pg.SelfClearDict()


def create_param_tuner(optimizer=pg.optimize):
    return pg.ParameterTuner(lambda params:
                              pg.Normalize(
                                  postprocessor(
                                      pg.GenericGraphFilter([1]+params,
                                                            preprocessor=pre,
                                                            error_type="iters",
                                                            max_iters=41,
                                                            optimization_dict=optimization,
                                                            preserve_norm=False))),
                             deviation_tol=1.E-6,
                             measure=measure,
                             optimizer=optimizer,
                             max_vals=[1]*40,
                             min_vals=[0]*40)


tuned = {
   "select": pg.AlgorithmSelection(algorithms.values(), fraction_of_training=0.9, measure=measure),
   "tune": create_param_tuner(),
   "tuneLBFGSB": create_param_tuner(pg.lbfgsb)
}

for name, graph, group in pg.load_datasets_all_communities(datasets, min_group_size=community_size, max_group_number=3):
    print(" & ".join([str(val) for val in [name, len(graph), graph.number_of_edges(), len(group)]])+" \\\\")

loader = pg.load_datasets_all_communities(datasets, min_group_size=community_size, max_group_number=3)
pg.benchmark_print(
    pg.benchmark_average((pg.benchmark(algorithms | tuned, loader, measure,
                                       fraction_of_training=[0.1, 0.2, 0.3], seed=list(range(1)))), posthocs=True),
    decimals=3, delimiter=" & ", end_line="\\\\")

