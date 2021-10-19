import pygrank as pg


datasets = ["amazon", "blockmodel", "citeseer", "eucore", "maven"]
#datasets = ["citeseer", "eucore", "maven"]
pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
tol = 1.E-9
optimization = pg.SelfClearDict()
algorithms = {
    "ppr0.85": pg.PageRank(alpha=0.85, preprocessor=pre, max_iters=10000, tol=tol),
    "ppr0.9": pg.PageRank(alpha=0.9, preprocessor=pre, max_iters=10000, tol=tol),
    "ppr0.99": pg.PageRank(alpha=0.99, preprocessor=pre, max_iters=10000, tol=tol),
    "hk3": pg.HeatKernel(t=3, preprocessor=pre, max_iters=10000, tol=tol, optimization_dict=optimization),
    "hk5": pg.HeatKernel(t=5, preprocessor=pre, max_iters=10000, tol=tol, optimization_dict=optimization),
    "hk7": pg.HeatKernel(t=7, preprocessor=pre, max_iters=10000, tol=tol, optimization_dict=optimization),
}
tuned = {
    "selected": pg.AlgorithmSelection(algorithms.values(), fraction_of_training=0.8),
    "tuned": pg.ParameterTuner(preprocessor=pre, fraction_of_training=0.8, tol=tol, optimization_dict=optimization, measure=pg.AUC),
    #"estimated": pg.HopTuner(preprocessor=pre, num_parameters=20, tol=tol, optimization_dict=optimization),
    "arnoldi": pg.HopTuner(preprocessor=pre, basis="arnoldi", num_parameters=20, tol=tol, optimization_dict=optimization, autoregression=5),
}

#algorithms = pg.create_variations(algorithms, {"": pg.Tautology, "+Sweep": pg.Sweep})
#print(algorithms.keys())

#for name, graph, group in pg.load_datasets_all_communities(datasets, min_group_size=50):
#    print(" & ".join([str(val) for val in [name, len(graph), graph.number_of_edges(), len(group)]])+" \\\\")
loader = pg.load_datasets_all_communities(datasets, min_group_size=50)
pg.benchmark_print(pg.benchmark(algorithms | tuned, loader, pg.AUC, fraction_of_training=.8, seed=list(range(1))),
                   decimals=3, delimiter=" & ", end_line="\\\\")

