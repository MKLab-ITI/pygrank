import pygrank as pg
from tensortune import Tensortune#, TensortuneOutputs


#datasets = ["acm", "amazon", "ant", "citeseer","dblp","facebook0","facebook686","log4j","maven","pubmed","squirel", "twitter"]
datasets = ["pokec"]
#datasets = ["citeseer", "ant", "eucore"]

pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
tol = 1.E-9
filter = pg.PageRank(alpha=0.85, preprocessor=pre, max_iters=10000, tol=tol)
#filter = pg.Normalize(filter)

algorithms = {"None": filter,
              "LFPRU": pg.LFPR(alpha=0.85, max_iters=10000, tol=tol, redistributor="uniform"),
              "LFPRP": pg.LFPR(alpha=0.85, max_iters=10000, tol=tol, redistributor="original"),
              "LFPRO": pg.AdHocFairness(filter, "O"),
              "Mult": pg.AdHocFairness(filter, "B"),
              #"FPers": pg.FairPersonalizer(filter, target_pRule=1, pRule_weight=1, max_residual=0, error_type=pg.Mabs, error_skewing=True, parity_type="impact"),
              #"Fest-C": pg.FairPersonalizer(filter, 1, pRule_weight=1, max_residual=1, error_type=pg.SpearmanCorrelation, error_skewing=True, parity_type="U"),
              #"Tensortune (gen)": Tensortune(filter, fix_personalization=True, fairness_weight=1),
              "Tensortune": Tensortune(filter, fix_personalization=True, fairness_weight=1, zero_mabs=0.01)#, max_fairness=0.8),
              }
#algorithms = pg.create_variations(algorithms, pg.Normalize)

pg.benchmark_print(pg.benchmark_average(pg.benchmark(algorithms,
                                pg.load_datasets_multiple_communities(datasets, max_group_number=100, directed=False),
                                metrics=pg.AUC,#[pg.Utility(pg.Mabs, filter)],
                                sensitive=lambda personalization, exclude: pg.pRule(personalization),
                                fraction_of_training=[2, .1, .5]#, seed=[0, 1, 2, 3]
                                )),
                   delimiter=" & ", end_line="\\\\", decimals=3)
