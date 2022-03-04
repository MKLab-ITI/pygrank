import pygrank as pg
from tensortune import Tensortune, LFPR, TensortuneOutputs


#datasets = ["acm", "amazon", "ant", "citeseer","dblp","facebook0","facebook686","log4j","maven","pubmed","squirel", "twitter"]
datasets = ["log4j", "citeseer", "facebook0", "ant", "eucore", "dblp"]

pre = pg.preprocessor(assume_immutability=True, normalization="col")
filter = pg.PageRank(alpha=0.85, preprocessor=pre, max_iters=10000, tol=1.E-9)

algorithms = {"None": filter,
              "LFPRU": LFPR(alpha=filter.alpha, max_iters=10000, tol=1.E-9, redistributor="uniform"),
              "LFPRP": LFPR(alpha=filter.alpha, max_iters=10000, tol=1.E-9, redistributor="original"),
              "Mult": pg.AdHocFairness(filter, "B"),
              "LFPRO": pg.AdHocFairness(filter, "O"),
              #"FPers-C": pg.FairPersonalizer(filter, .8, pRule_weight=10, max_residual=0, error_type=pg.L2, error_skewing=True, parity_type="impact"),
              #"Fest-C": pg.FairPersonalizer(filter, .8, pRule_weight=10, max_residual=1, error_type=pg.L2, error_skewing=False, parity_type="impact"),
              "Tensortune": Tensortune(filter),
              #"TensortuneOutputs": TensortuneOutputs(filter),
              #"TensortuneCombineed": TensortuneOutputs(Tensortune(filter), filter)
              }
algorithms = pg.create_variations(algorithms, pg.Normalize)

pg.benchmark_print(pg.benchmark(algorithms, pg.load_datasets_multiple_communities(datasets, max_group_number=2),
                                metric=pg.AUC, sensitive=pg.pRule, fraction_of_training=[0.1]),
                   delimiter=" & ", end_line="\\\\")