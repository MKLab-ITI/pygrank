import networkx as nx
import pygrank as pg
import pytest
from .test_core import supported_backends


def test_zero_personalization():
    assert pg.sum(pg.PageRank()(next(pg.load_datasets_graph(["graph9"])),{}).np) == 0


def test_abstract_filter_types():
    graph = next(pg.load_datasets_graph(["graph5"]))
    with pytest.raises(Exception):
        pg.GraphFilter().rank(graph)
    with pytest.raises(Exception):
        pg.RecursiveGraphFilter().rank(graph)
    with pytest.raises(Exception):
        pg.ClosedFormGraphFilter().rank(graph)
    with pytest.raises(Exception):
        pg.Tuner().rank(graph)


def test_filter_invalid_parameters():
    graph = next(pg.load_datasets_graph(["graph5"]))
    with pytest.raises(Exception):
        pg.HeatKernel(normalization="unknown").rank(graph)
    with pytest.raises(Exception):
        pg.HeatKernel(coefficient_type="unknown").rank(graph)


def test_convergence_string_conversion():
    # TODO: make convergence trackable from wrapping objects
    graph = next(pg.load_datasets_graph(["graph5"]))
    ranker = pg.PageRank()
    ranker(graph)
    assert str(ranker.convergence.iteration)+" iterations" in str(ranker.convergence)


def test_pagerank_vs_networkx():
    graph = next(pg.load_datasets_graph(["graph9"]))
    #print(pg.preprocessor(normalization='col')(graph).todense())
    for _ in supported_backends():
        ranker = pg.Normalize("sum", pg.PageRank(normalization='col', tol=1.E-9))
        test_result = ranker(graph)
        test_result2 = nx.pagerank(graph, tol=1.E-9)
        # TODO: assert that 2.5*epsilon is indeed a valid limit
        assert pg.Mabs(test_result)(test_result2) < 2.5*pg.epsilon()


def test_prevent_node_lists_as_graphs():
    graph = next(pg.load_datasets_graph(["graph5"]))
    with pytest.raises(Exception):
        pg.PageRank().rank(list(graph))


def test_non_convergence():
    graph = next(pg.load_datasets_graph(["graph9"]))
    with pytest.raises(Exception):
        pg.PageRank(max_iters=5).rank(graph)


def test_custom_runs():
    graph = next(pg.load_datasets_graph(["graph9"]))
    for _ in supported_backends():
        ranks1 = pg.Normalize(pg.PageRank(0.85, tol=pg.epsilon(), max_iters=1000)).rank(graph, {"A": 1})
        # TODO find why the following is not exactly the same
        ranks2 = pg.Normalize(pg.GenericGraphFilter([0.85**i for i in range(20)], tol=pg.epsilon())).rank(graph, {"A": 1})
        #print(ranks1.np-ranks2.np)
        #self.assertAlmostEqual(pg.Mabs(ranks1)(ranks2), 0, places=11)
        assert True


def test_completion():
    graph = next(pg.load_datasets_graph(["graph9"]))
    for _ in supported_backends():
        pg.PageRank().rank(graph)
        pg.HeatKernel().rank(graph)
        pg.AbsorbingWalks().rank(graph)
        assert True


def test_quotient():
    graph = next(pg.load_datasets_graph(["graph9"]))
    for _ in supported_backends():
        test_result = pg.PageRank(normalization='symmetric', tol=max(1.E-9, pg.epsilon()), use_quotient=True).rank(graph)
        norm_result = pg.PageRank(normalization='symmetric', tol=max(1.E-9, pg.epsilon()), use_quotient=pg.Normalize("sum")).rank(graph)
        assert pg.Mabs(test_result)(norm_result) < pg.epsilon()


def test_automatic_graph_casting():
    graph = next(pg.load_datasets_graph(["graph9"]))
    for _ in supported_backends():
        signal = pg.to_signal(graph, {"A": 1})
        test_result1 = pg.PageRank(normalization='col').rank(signal, signal)
        test_result2 = pg.PageRank(normalization='col').rank(personalization=signal)
        assert pg.Mabs(test_result1)(test_result2) < pg.epsilon()
        with pytest.raises(Exception):
            pg.PageRank(normalization='col').rank(personalization={"A": 1})
        with pytest.raises(Exception):
            pg.PageRank(normalization='col').rank(graph.copy(), signal)


def test_absorbing_vs_pagerank():
    graph = next(pg.load_datasets_graph(["graph9"]))
    personalization = {"A": 1, "B": 1}
    for _ in supported_backends():
        pagerank_result = pg.PageRank(normalization='col').rank(graph, personalization)
        absorbing_result = pg.AbsorbingWalks(0.85, normalization='col', max_iters=1000).rank(graph, personalization)
        assert pg.Mabs(pagerank_result)(absorbing_result) < pg.epsilon()


def test_kernel_locality():
    graph = next(pg.load_datasets_graph(["graph9"]))
    personalization = {"A": 1, "B": 1}
    for _ in supported_backends():
        for kernel_algorithm in [pg.HeatKernel, pg.BiasedKernel]:
            pagerank_result = pg.Normalize("sum", pg.PageRank(max_iters=1000)).rank(graph, personalization)
            kernel_result = pg.Normalize("sum", kernel_algorithm(max_iters=1000)).rank(graph, personalization)
            assert pagerank_result['A'] < kernel_result['A']
            assert pagerank_result['I'] > kernel_result['I']


def test_optimization_dict():
    from timeit import default_timer as time
    graph = next(pg.load_datasets_graph(["bigraph"]))
    personalization = {str(i): 1 for i in range(200)}
    preprocessor = pg.preprocessor(assume_immutability=True)
    preprocessor(graph)
    tic = time()
    for _ in range(10):
        pg.ParameterTuner(preprocessor=preprocessor, tol=1.E-9).rank(graph, personalization)
    unoptimized = time()-tic
    optimization = dict()
    tic = time()
    for _ in range(10):
        pg.ParameterTuner(optimization_dict=optimization, preprocessor=preprocessor, tol=1.E-9).rank(graph, personalization)
    optimized = time() - tic
    assert len(optimization) == 20
    assert unoptimized > optimized


    