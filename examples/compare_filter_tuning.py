import pygrank as pg

pg.load_backend("matvec")

#datasets = ["amazon", "citeseer", "maven"]
datasets = ["amazon"]
community_size = 500

pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
convergence = {"tol": 1.E-9, "max_iters": 10000}
#convergence = {"error_type": "iters", "max_iters": 20}
tuned = [1]+[10000, 0.7573524630180889, 0.0, 1.0, 0.004950495049504955, 0.4975492598764827, 0.2573767277717871, 0.0, 0.2549259876482698, 0.009851975296539583, 0.5, 0.5, 0.5, 0.2549259876482698, 0.5, 0.5, 0.5, 0.009851975296539583, 0.009851975296539583, 0.25002450740123516, 0.2524752475247525, 0.0, 0.5, 0.0, 0.009851975296539583, 0.0, 0.004950495049504955, 0.0, 0.5, 0.004950495049504955, 0.7475247524752475, 0.004950495049504955, 0.009851975296539583, 0.0, 0.995049504950495, 0.0, 0.5, 0.0, 0.004950495049504955, 0.0]

algorithms = {
    "ppr0.5": pg.PageRank(alpha=0.5, preprocessor=pre, **convergence),
    "ppr0.85": pg.PageRank(alpha=0.85, preprocessor=pre, **convergence),
    "ppr0.9": pg.PageRank(alpha=0.9, preprocessor=pre, **convergence),
    "ppr0.99": pg.PageRank(alpha=0.99, preprocessor=pre, **convergence),
    "hk2": pg.HeatKernel(t=2, preprocessor=pre, **convergence),
    "hk3": pg.HeatKernel(t=3, preprocessor=pre, **convergence),
    "hk5": pg.HeatKernel(t=5, preprocessor=pre, **convergence),
    "hk7": pg.HeatKernel(t=7, preprocessor=pre, **convergence),
    #"exp": pg.GenericGraphFilter(tuned, preprocessor=pre, error_type="iters", max_iters=41)
}

postprocessor = pg.Tautology
algorithms = pg.benchmarks.create_variations(algorithms, postprocessor)
measure = pg.Time
optimization = pg.SelfClearDict()


def create_param_tuner(optimizer=pg.optimize):
    return pg.ParameterTuner(lambda params:
                      pg.Normalize(postprocessor(
                          pg.GenericGraphFilter([1]+params, preprocessor=pre, error_type="iters", max_iters=41,
                                                optimization_dict=optimization, preserve_norm=False))),
                      deviation_tol=1.E-6, measure=measure, optimizer=optimizer, max_vals=[1]*40, min_vals=[0]*40)



tuned = {
   "select": pg.AlgorithmSelection(algorithms.values()),#, combined_prediction=False),
   #"tune": create_param_tuner(),
   #"tuneLBFGSB": create_param_tuner(pg.lbfgsb)
}

for name, graph, group in pg.load_datasets_all_communities(datasets, min_group_size=community_size, max_group_number=3):
    print(" & ".join([str(val) for val in [name, len(graph), graph.number_of_edges(), len(group)]])+" \\\\")

loader = pg.load_datasets_all_communities(datasets, min_group_size=community_size, max_group_number=3)
pg.benchmark_print(
    pg.benchmark_average((pg.benchmark(algorithms | tuned, loader, measure,
                   fraction_of_training=[0.1, 0.2, 0.3], seed=list(range(1)))), posthocs=True),
                   decimals=3, delimiter=" & ", end_line="\\\\")

