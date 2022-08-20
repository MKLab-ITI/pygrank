import pygrank as pg
import io
from .test_core import supported_backends


def test_benchmark_print():
    assert pg.benchmarks.printers._fraction2str(0.1) == ".10"
    assert pg.benchmarks.printers._fraction2str(0.00001) == "0"
    assert pg.benchmarks.printers._fraction2str(1) == "1.00"
    loader = pg.load_datasets_one_community(["graph9", "bigraph"])
    console = pg.benchmark_print(pg.benchmark(pg.create_demo_filters(), loader),
                                 out=io.StringIO(""), err=None).getvalue()
    loader = pg.load_datasets_one_community(["graph9", "bigraph"])
    ret = pg.benchmark_dict(pg.benchmark(pg.create_demo_filters(), loader, sensitive=pg.MannWhitneyParity))
    assert isinstance(ret, dict)
    assert len(ret) == 3
    assert isinstance(ret["graph9"], dict)
    assert (len(str(ret)) - len(console)) < (len(str(ret)) + len(console))/2


def test_algorithm_selection():
    for _ in supported_backends():
        _, graph, communities = next(pg.load_datasets_multiple_communities(["bigraph"], max_group_number=3))
        train, test = pg.split(communities, 0.05)  # 5% of community members are known
        algorithms = pg.create_variations(pg.create_demo_filters(), pg.Normalize)

        supervised_algorithm = pg.AlgorithmSelection(algorithms.values(), measure=pg.AUC, tuning_backend="numpy")
        print(supervised_algorithm.cite())
        modularity_algorithm = pg.AlgorithmSelection(algorithms.values(), fraction_of_training=1,
                                                     measure=pg.Modularity().as_supervised_method(), tuning_backend="numpy")

        supervised_aucs = list()
        modularity_aucs = list()
        for seeds, members in zip(train.values(), test.values()):
            measure = pg.AUC(members, exclude=seeds)
            supervised_aucs.append(measure(supervised_algorithm(graph, seeds)))
            modularity_aucs.append(measure(modularity_algorithm(graph, seeds)))

        assert abs(sum(supervised_aucs) / len(supervised_aucs) - sum(modularity_aucs) / len(modularity_aucs)) < 0.05


def test_unsupervised_vs_auc():
    def loader():
        return pg.load_datasets_multiple_communities(["graph9"])

    algorithms = pg.create_variations(pg.create_many_filters(), pg.create_many_variation_types())
    time_scores = pg.benchmark_scores(pg.benchmark(algorithms, loader(), pg.Time))
    assert sum(time_scores) > 0

    measures = {"AUC": lambda ground_truth, exclude: pg.MultiSupervised(pg.AUC, ground_truth, exclude),
                "NDCG": lambda ground_truth, exclude: pg.MultiSupervised(pg.NDCG, ground_truth, exclude),
                "Density": lambda graph: pg.MultiUnsupervised(pg.Density, graph),
                "Conductance": lambda graph: pg.MultiUnsupervised(pg.Conductance(autofix=True).as_unsupervised_method(), graph),
                "Modularity": lambda graph: pg.MultiUnsupervised(pg.Modularity(max_positive_samples=5).as_unsupervised_method(), graph),
                "CCcos": lambda graph: pg.ClusteringCoefficient(graph, similarity="cos", max_positive_samples=5),
                "CCdot": lambda graph: pg.ClusteringCoefficient(graph, similarity="dot", max_positive_samples=5),
                "LinkAUCcos": lambda graph: pg.LinkAssessment(graph, similarity="cos", max_positive_samples=5),
                "LinkAUCdot": lambda graph: pg.LinkAssessment(graph, similarity="dot", max_positive_samples=5),
                "HopAUCcos": lambda graph: pg.LinkAssessment(graph, similarity="cos", hops=2, max_positive_samples=5),
                "HopAUCdot": lambda graph: pg.LinkAssessment(graph, similarity="dot", hops=2, max_positive_samples=5),
                }

    scores = {}#measure: pg.benchmark_scores(pg.benchmark(algorithms, loader(), measures[measure])) for measure in measures}
    for measure in measures:  # do this as a for loop, because pytest becomes a little slow above list comprehension
        print(measure)
        scores[measure] = pg.benchmark_scores(pg.benchmark(algorithms, loader(), measures[measure]))
    supervised = {"AUC", "NDCG"}
    evaluations = dict()
    for measure in measures:
        evaluations[measure] = abs(pg.SpearmanCorrelation(scores["AUC"])(scores[measure]))
    #for measure in measures:
    #    print(measure, evaluations[measure])
    assert max([evaluations[measure] for measure in measures if measure not in supervised]) == evaluations["LinkAUCdot"]


def test_one_community_benchmarks():
    pg.load_backend("numpy")
    datasets = ["graph9", "bigraph"]
    pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
    algorithms = {
        "ppr0.85": pg.PageRank(alpha=0.85, preprocessor=pre, max_iters=10000, tol=1.E-9),
        "ppr0.99": pg.PageRank(alpha=0.99, preprocessor=pre, max_iters=10000, tol=1.E-9),
        "hk3": pg.HeatKernel(t=3, preprocessor=pre, max_iters=10000, tol=1.E-9),
        "hk5": pg.HeatKernel(t=5, preprocessor=pre, max_iters=10000, tol=1.E-9),
        "tuned": pg.ParameterTuner(preprocessor=pre, max_iters=10000, tol=1.E-9),
    }
    # algorithms = benchmark.create_variations(algorithms, {"": pg.Tautology, "+SO": pg.SeedOversampling})
    # loader = pg.load_datasets_one_community(datasets)
    # pg.benchmark(algorithms, loader, "time", verbose=True)

    loader = pg.load_datasets_one_community(datasets)
    pg.benchmark_print(pg.benchmark_average(pg.benchmark_ranks(pg.benchmark(algorithms, loader, pg.AUC, fraction_of_training=.8))))


def test_load_dataset_load():
    assert len(list(pg.load_datasets_all_communities(["graph9"]))) > 1


def test_dataset_lists():
    assert len(pg.downloadable_small_datasets()) > 1
    assert len(pg.downloadable_small_datasets()) < len(pg.datasets)
    assert len(pg.downloadable_datasets()) == len(pg.datasets)


def test_all_communities_benchmarks():
    datasets = ["bigraph"]
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

    tuned = {"selected": pg.AlgorithmSelection(algorithms.values(), fraction_of_training=0.8)}
    loader = pg.load_datasets_all_communities(datasets, min_group_size=50)
    pg.benchmark_print(pg.benchmark(algorithms | tuned, loader, pg.AUC, fraction_of_training=.8, seed=list(range(1))),
                       decimals=3, delimiter=" & ", end_line="\\\\")
    loader = pg.load_datasets_all_communities(datasets, min_group_size=50)
    pg.benchmark_print(pg.benchmark(algorithms | tuned, loader, pg.Modularity, sensitive=pg.pRule, fraction_of_training=.8, seed=list(range(1))),
                       decimals=3, delimiter=" & ", end_line="\\\\")
    mistreatment = lambda known_scores, sensitive_signal, exclude: \
        pg.AM([pg.Disparity([pg.TPR(known_scores, exclude=1 - (1 - exclude.np) * sensitive_signal.np),
                             pg.TPR(known_scores, exclude=1 - (1 - exclude.np) * (1 - sensitive_signal.np))]),
               pg.Disparity([pg.TNR(known_scores, exclude=1 - (1 - exclude.np) * sensitive_signal.np),
                             pg.TNR(known_scores, exclude=1 - (1 - exclude.np) * (1 - sensitive_signal.np))])])
    loader = pg.load_datasets_all_communities(datasets, min_group_size=50)
    pg.benchmark_print(pg.benchmark(algorithms | tuned, loader, pg.Modularity, sensitive=mistreatment, fraction_of_training=.8, seed=list(range(1))),
                       decimals=3, delimiter=" & ", end_line="\\\\")


def test_multigroup_benchmarks():
    datasets = ["bigraph"]
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

    tuned = {"selected": pg.AlgorithmSelection(algorithms.values(), fraction_of_training=0.8)}
    loader = pg.load_datasets_multiple_communities(datasets, min_group_size=50)
    pg.benchmark_print(pg.benchmark(algorithms | tuned, loader, lambda ground_truth, exclude: pg.MultiSupervised(pg.AUC, ground_truth, exclude), fraction_of_training=.8, seed=list(range(1))),
                       decimals=3, delimiter=" & ", end_line="\\\\")
    loader = pg.load_datasets_multiple_communities(datasets, min_group_size=50)
    pg.benchmark_print(pg.benchmark(algorithms | tuned, loader, pg.Modularity, sensitive=pg.pRule, fraction_of_training=.8, seed=list(range(1))),
                       decimals=3, delimiter=" & ", end_line="\\\\")

