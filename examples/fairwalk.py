import pygrank as pg

#datasets = ["acm", "amazon", "ant", "citeseer","dblp","facebook0","facebook686","log4j","maven","pubmed","squirel", "twitter"]
datasets = ["facebook0","facebook686", "log4j", "ant", "eucore", "citeseer", "dblp"]
seed_fractions = [0.3]
pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")

filters = {
    "ppr0.85": pg.PageRank(alpha=0.85, preprocessor=pre, max_iters=10000, tol=1.E-9),
    "ppr0.99": pg.PageRank(alpha=0.99, preprocessor=pre, max_iters=10000, tol=1.E-9),
    "hk3": pg.HeatKernel(t=3, preprocessor=pre, max_iters=10000, tol=1.E-9),
    "hk7": pg.HeatKernel(t=7, preprocessor=pre, max_iters=10000, tol=1.E-9),
}
filters = pg.create_variations(filters, {"": pg.Tautology, "+Sweep": pg.Sweep})

for name, filter in filters.items():
    print("=====", name, "=====")
    algorithms = {"None": filter,
                  "Mult": pg.AdHocFairness(filter, "B"),
                  "LFPRO": pg.AdHocFairness(filter, "O"),
                  "FBuck-C": pg.FairPersonalizer(filter, .8, pRule_weight=10, max_residual=1, error_type=pg.Mabs, parameter_buckets=0),
                  "FPers-C": pg.FairPersonalizer(filter, .8, pRule_weight=10, max_residual=0, error_type=pg.Mabs, error_skewing=True),
                  "Fest-C": pg.FairPersonalizer(filter, .8, pRule_weight=10, max_residual=1, error_type=pg.Mabs)
                 }
    algorithms = pg.create_variations(algorithms, {"": pg.Normalize})

    #import cProfile as profile
    #pr = profile.Profile()
    #pr.enable()
    pg.benchmark_print(pg.benchmark(algorithms, pg.load_datasets_multiple_communities(datasets, max_group_number=2),
                                    metric=pg.AUC, sensitive=pg.pRule, fraction_of_training=seed_fractions),
                       delimiter=" & ", end_line="\\\\")

    #pr.disable()
    #pr.dump_stats('profile.pstat')