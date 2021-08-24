import pygrank as pg
def test_benchmark_print(self):
    self.assertEqual(pg.benchmark.utils._fraction2str(0.1), ".10")
    self.assertEqual(pg.benchmark.utils._fraction2str(0.00001), "0")
    self.assertEqual(pg.benchmark.utils._fraction2str(1), "1.00")
    pg.benchmark_print(pg.supervised_benchmark(pg.create_demo_filters(), pg.load_datasets_one_community(["ant"])))
    ret = pg.benchmark_dict(pg.supervised_benchmark(pg.create_demo_filters(), pg.load_datasets_one_community(["ant"])))
    self.assertTrue(isinstance(ret, dict))
    self.assertTrue(isinstance(ret["ant"], dict))


def test_unsupervised_vs_auc(self):
    algorithms = pg.create_variations(pg.create_many_filters(), pg.create_many_variation_types())
    auc_scores = pg.benchmark_scores(pg.supervised_benchmark(algorithms, pg.load_datasets_one_community(["ant"]), pg.AUC))
    self.assertGreater(sum(pg.benchmark_scores(pg.supervised_benchmark(algorithms, pg.load_datasets_one_community(["ant"]), "time"))), 0)
    conductance_scores = pg.benchmark_scores(pg.supervised_benchmark(algorithms, pg.load_datasets_one_community(["ant"]),
                                                                     lambda _, __: pg.Conductance()))
    density_scores = pg.benchmark_scores(pg.supervised_benchmark(algorithms, pg.load_datasets_one_community(["ant"]),
                                                                 lambda _, __: pg.Density()))
    modularity_scores = pg.benchmark_scores(pg.supervised_benchmark(algorithms, pg.load_datasets_one_community(["ant"]),
                                                                    lambda _, __: pg.Modularity(max_positive_samples=100)))
    pg.PearsonCorrelation(auc_scores)(conductance_scores)
    pg.SpearmanCorrelation(auc_scores)(density_scores)
    pg.SpearmanCorrelation(auc_scores)(modularity_scores)


def test_benchmarks(self):
    datasets = pg.downloadable_small_datasets()
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
    # pg.supervised_benchmark(algorithms, loader, "time", verbose=True)

    loader = pg.load_datasets_one_community(datasets)
    pg.benchmark_print(pg.supervised_benchmark(algorithms, loader, pg.AUC, fraction_of_training=.8))
    loader = pg.load_datasets_one_community(datasets)
    pg.benchmark_print(pg.unsupervised_benchmark(algorithms, loader, pg.Conductance, fraction_of_training=.8))



def test_load_datasets_all_communities(self):
    self.assertGreater(len(list(pg.load_datasets_all_communities(["citeseer"]))), 1)

def test_dataset_generation(self):
    self.assertEquals(len(pg.downloadable_datasets()), len(pg.datasets))

    def test_multigroup(self):
        datasets = pg.downloadable_small_datasets()
        pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
        algorithms = {
            "ppr0.85": pg.PageRank(alpha=0.85, preprocessor=pre, max_iters=10000, tol=1.E-9),
            "ppr0.99": pg.PageRank(alpha=0.99, preprocessor=pre, max_iters=10000, tol=1.E-9),
            "hk3": pg.HeatKernel(t=3, preprocessor=pre, max_iters=10000, tol=1.E-9),
            "hk5": pg.HeatKernel(t=5, preprocessor=pre, max_iters=10000, tol=1.E-9),
        }
        loader = pg.load_datasets_multiple_communities(datasets)
        pg.benchmark_print(pg.supervised_benchmark(algorithms, loader,
                                                   lambda ground_truth, exlude: pg.MultiSupervised(pg.AUC, ground_truth, exlude)))
        loader = pg.load_datasets_multiple_communities(datasets)
        pg.benchmark_print(pg.unsupervised_benchmark(algorithms, loader,
                                                   lambda graph: pg.LinkAssessment(graph, max_positive_samples=200, max_negative_samples=200)))
        loader = pg.load_datasets_multiple_communities(datasets)
        pg.benchmark_print(pg.unsupervised_benchmark(algorithms, loader,
                                                   lambda graph: pg.LinkAssessment(graph, hops=2, similarity="dot", max_positive_samples=200, max_negative_samples=200)))
        loader = pg.load_datasets_multiple_communities(datasets)
        pg.benchmark_print(pg.unsupervised_benchmark(algorithms, loader,
                                                   lambda graph: pg.ClusteringCoefficient(graph)))
        loader = pg.load_datasets_multiple_communities(datasets)
        pg.benchmark_print(pg.unsupervised_benchmark(algorithms, loader,
                                                   lambda graph: pg.ClusteringCoefficient(graph, similarity="dot")))
        loader = pg.load_datasets_multiple_communities(datasets)
        pg.benchmark_print(pg.unsupervised_benchmark(algorithms, loader,
                                                   lambda graph: pg.MultiUnsupervised(pg.Conductance, graph)))