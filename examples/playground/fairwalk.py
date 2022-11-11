import pygrank as pg
from tensortune import Tensortune, TensortuneOutputs


#datasets = ["acm", "amazon", "ant", "citeseer","dblp","facebook0","facebook686","log4j","maven","pubmed","squirel", "twitter"]
datasets = ["amazon", "eucore", "citeseer"]#["amazon", "log4j", "citeseer", "facebook100", "ant", "eucore"]

pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
tol = 1.E-9
filter = pg.PageRank(alpha=0.85, preprocessor=pre, max_iters=10000, tol=tol)

algorithms = {"None": filter,
              #"LFPRU": pg.LFPR(alpha=filter.alpha, max_iters=10000, tol=tol, redistributor="uniform"),
              #"LFPRP": pg.LFPR(alpha=filter.alpha, max_iters=10000, tol=tol, redistributor="original"),
              "Mult": pg.AdHocFairness(filter, "B"),
              "LFPRO": pg.AdHocFairness(filter, "O"),
              "FPers-C": pg.FairPersonalizer(filter, .8, pRule_weight=10, max_residual=0,
                                             error_type=pg.Mabs, error_skewing=True, parity_type="impact"),
              #"Fest-C": pg.FairPersonalizer(filter, 1, pRule_weight=1, max_residual=1,
              #                              error_type=pg.SpearmanCorrelation, error_skewing=True, parity_type="U"),
              "Tensortune": Tensortune(filter, robustness=0.081),
              #"TensortuneOutputs": TensortuneOutputs(filter),
              #"TensortuneCombineed": TensortuneOutputs(Tensortune(filter), filter)
              }
algorithms = pg.create_variations(algorithms, pg.Normalize)

pg.benchmark_print(pg.benchmark(algorithms, pg.load_datasets_multiple_communities(datasets, max_group_number=2,directed=False),
                                metric=pg.AUC, sensitive=pg.pRule, fraction_of_training=[0.3, 0.5]),
                   delimiter=" & ", end_line="\\\\")