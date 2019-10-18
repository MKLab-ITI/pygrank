import unittest
import networkx as nx
import time


def create_test_graph(directed=False):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "D")
    G.add_edge("E", "F")
    G.add_edge("F", "G")
    G.add_edge("G", "H")
    G.add_edge("H", "I")
    G.add_edge("I", "J")
    G.add_edge("J", "K")
    G.add_edge("A", "D")
    G.add_edge("B", "D")
    G.add_edge("B", "E")
    G.add_edge("E", "G")
    G.add_edge("G", "J")
    G.add_edge("G", "I")
    G.add_edge("H", "J")
    G.add_edge("I", "K")
    G.add_edge("L", "K")
    G.add_edge("K", "M")
    return G


class Test(unittest.TestCase):
    def test_rank_results(self):
        from pygrank.algorithms.pagerank import PageRank as Ranker
        G = create_test_graph()
        test_result = Ranker(normalization='col').rank(G)
        nx_result = nx.pagerank_scipy(G)
        abs_diffs = sum(abs(test_result[v]-nx_result[v]) for v in nx_result.keys())/len(nx_result)
        self.assertAlmostEqual(abs_diffs, 0, places=16, msg="PageRank compliance with nx results")

    def test_directed_rank_results(self):
        from pygrank.algorithms.pagerank import PageRank as Ranker
        G = create_test_graph(directed=True)
        test_result = Ranker(normalization='col').rank(G)
        nx_result = nx.pagerank_scipy(G)
        abs_diffs = sum(abs(test_result[v] - nx_result[v]) for v in nx_result.keys()) / len(nx_result)
        self.assertAlmostEqual(abs_diffs, 0, places=16, msg="DirectedPageRank compliance with nx results")

    def test_rank_time(self):
        from pygrank.algorithms.pagerank import PageRank as ranker
        import scipy.stats
        nx_time = list()
        test_time = list()
        repeats = 50
        for _ in range(repeats):
            G = create_test_graph()
            tic = time.clock()
            ranker(normalization='col').rank(G)
            test_time.append(time.clock()-tic)
            tic = time.clock()
            nx.pagerank_scipy(G)
            nx_time.append(time.clock()-tic)
        self.assertLessEqual(scipy.stats.ttest_ind(nx_time, test_time)[1], 0.001, msg="PageRank time comparable to nx with p-value<0.001")

    def test_reduce_running_time(self):
        from pygrank.algorithms.pagerank import PageRank as ranker
        from pygrank.algorithms.pagerank import Fast as fast
        import scipy.stats
        test_time = list()
        dynamic_time = list()
        repeats = 1
        personalization = {"A": 1, "B": 1}
        abs_diffs = list()
        for _ in range(repeats):
            G = create_test_graph()
            tic = time.clock()
            test_result = ranker(normalization='col', alpha=0.99).rank(G, personalization)
            test_time.append(time.clock()-tic)
            tic = time.clock()
            dynamic_result = fast(ranker(normalization='col', alpha=0.99)).rank(G, personalization)
            dynamic_time.append(time.clock()-tic)
            print(sum(test_time), sum(dynamic_time))
            abs_diffs.append(sum(abs(test_result[v] - dynamic_result[v]) for v in dynamic_result.keys()) / len(dynamic_result))
            #different_result = fast(ranker(normalization='col', alpha=0.85)).rank(G, personalization)
            #print("Should be different", sum(abs(test_result[v] - different_result[v]) for v in different_result.keys()) / len(different_result))
        self.assertLessEqual(sum(dynamic_time), sum(test_time), msg="Fast dynamic parameter faster than PageRank")
        self.assertAlmostEqual(sum(abs_diffs)/len(abs_diffs), 0, places=16, msg="Fast dynamic parameter convergence compliance to PageRank")

    def test_symmetric_normalization_symmetricity(self):
        from pygrank.algorithms.utils import to_scipy_sparse_matrix
        G = create_test_graph(directed=False)
        M = to_scipy_sparse_matrix(G, "symmetric").todense()
        for i in range(len(G)):
            for j in range(len(G)):
                self.assertEqual(M[i,j], M[j,i], msg="Symmetricity of symmetric normalization")

    def test_heat_kernel_locality(self):
        from pygrank.algorithms.pagerank import PageRank
        from pygrank.algorithms.pagerank import HeatKernel
        G = create_test_graph()
        personalization = {"A": 1, "B": 1}
        pagerank = PageRank().rank(G, personalization)
        heatkernel = HeatKernel().rank(G, personalization)
        self.assertLess(pagerank['D']/sum(pagerank.values()), heatkernel['D']/sum(heatkernel.values()), msg="HeatKernel more local than PageRank")
        self.assertLess(heatkernel['I']/sum(heatkernel.values()), pagerank['I']/sum(pagerank.values()), msg="HeatKernel more local than PageRank")

    def test_oversampling_importance(self):
        from pygrank.algorithms.pagerank import PageRank as Ranker
        from pygrank.algorithms.oversampling import SeedOversampling as Oversampler
        G = create_test_graph()
        personalization = {"A": 1, "B": 1}
        ranks = Ranker().rank(G, personalization)
        oversampled = Oversampler(Ranker()).rank(G, personalization)
        self.assertLess(oversampled['A'], ranks['A'], msg="Oversampling affects ranks")

    def test_oversampling_importance(self):
        from pygrank.algorithms.pagerank import PageRank as Ranker
        from pygrank.algorithms.oversampling import SeedOversampling as Oversampler
        from pygrank.algorithms.oversampling import BoostedSeedOversampling as BoostedOversampler
        G = create_test_graph()
        personalization = {"A": 1, "B": 1}
        oversampled = Oversampler(Ranker()).rank(G, personalization)
        boosted_oversampled = BoostedOversampler(Ranker()).rank(G, personalization)
        # need to assert 5 places precision since default tol=1.E-6
        self.assertAlmostEqual(boosted_oversampled['B']/boosted_oversampled['A'], oversampled['B']/oversampled['A'], places=5, msg="Boosting ranks can find relative oversampling ranks")

    def test_ndcg(self):
        from pygrank.metrics.supervised import NDCG
        ndcg = NDCG({"Doc1":3, "Doc2":2, "Doc3":3, "Doc4":0, "Doc5":1, "Doc6":2})
        score = ndcg.evaluate({"Doc1":6, "Doc2":5, "Doc3":4, "Doc4":3, "Doc5":2, "Doc6":1})
        self.assertAlmostEqual(score, 0.961, places=3) # wikipedia example

if __name__ == '__main__':
    unittest.main()
