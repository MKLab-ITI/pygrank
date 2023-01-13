import pygrank as pg
from tensortune import Tensortune


def experiment(filter, datasets, task_metric, fraction_of_training, fix_personalization=True, repeats=1, sensitive_group=100):
    if hasattr(filter, "alpha"):
        algorithms = {"None": filter,
                      "LFPRU": pg.LFPR(alpha=filter.alpha, max_iters=10000, tol=filter.convergence.tol, redistributor="uniform",  fix_personalization=fix_personalization),
                      "LFPRP": pg.LFPR(alpha=filter.alpha, max_iters=10000, tol=filter.convergence.tol, redistributor="original",  fix_personalization=fix_personalization),
                      "LFPRO": pg.AdHocFairness(filter, "O"),
                      "Mult": pg.AdHocFairness(filter, "B"),
                      #"FPers": pg.FairPersonalizer(filter,  fix_personalization=fix_personalization),
                      "Tensortune": Tensortune(filter, fix_personalization=fix_personalization)
                      }
    else:
        algorithms = {"None": filter,
                      "LFPRO": pg.AdHocFairness(filter, "O"),
                      "Mult": pg.AdHocFairness(filter, "B"),
                      "FPers": pg.FairPersonalizer(filter,  fix_personalization=fix_personalization),
                      "Tensortune": Tensortune(filter, fix_personalization=fix_personalization, fairness_weight=1)
                      }

    pg.benchmark_print(pg.benchmark_average(pg.benchmark(algorithms,
                                                         pg.load_datasets_multiple_communities(datasets,
                                                                                               max_group_number=sensitive_group,
                                                                                               directed=False),
                                                         metrics=task_metric
                                                            if not fix_personalization
                                                            else lambda personalization, exclude: task_metric(personalization),  # [pg.Utility(pg.Mabs, filter)],
                                                         sensitive=pg.pRule if not fix_personalization
                                                            else lambda personalization, exclude: pg.pRule(personalization),
                                                         fraction_of_training=fraction_of_training,
                                                         seed=list(range(repeats))
                                                         )),
                       delimiter=" & ", end_line="\\\\", decimals=3)