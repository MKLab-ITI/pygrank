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
        from pygrank.algorithms.utils import preprocessor
        G = create_test_graph()
        test_result = Ranker(to_scipy=preprocessor('col')).rank(G)
        nx_result = nx.pagerank_scipy(G)
        abs_diffs = sum(abs(test_result[v]-nx_result[v]) for v in nx_result.keys())/len(nx_result)
        self.assertAlmostEqual(abs_diffs, 0, places=16, msg="PageRank compliance with nx results")

    def test_rank_results(self):
        from pygrank.algorithms.pagerank import PageRank as Ranker1
        from pygrank.algorithms.filters import LanczosFilter as Ranker2
        G = create_test_graph()
        ranker1 = Ranker1(normalization="symmetric", use_quotient=False)
        test_result1 = ranker1.rank(G)
        ranker2 = Ranker2(normalization="symmetric", weights=[(1-ranker1.alpha)*ranker1.alpha**n for n in range(ranker1.convergence.iteration*30)], krylov_space_degree=5)
        test_result2 = ranker2.rank(G)
        abs_diffs = sum(abs(test_result1[v] - test_result2[v]) for v in test_result1.keys()) / len(test_result1)
        self.assertAlmostEqual(abs_diffs, 0, places=2, msg="Krylov space analysis compliance with PageRank")

    def test_rank_time(self):
        from pygrank.algorithms.pagerank import PageRank as ranker
        from pygrank.algorithms.utils import preprocessor
        import scipy.stats
        nx_time = list()
        test_time = list()
        repeats = 50
        for _ in range(repeats):
            G = create_test_graph()
            tic = time.clock()
            ranker(to_scipy=preprocessor('col')).rank(G)
            test_time.append(time.clock()-tic)
            tic = time.clock()
            nx.pagerank_scipy(G)
            nx_time.append(time.clock()-tic)
        self.assertLessEqual(scipy.stats.ttest_ind(nx_time, test_time)[1], 0.001, msg="PageRank time comparable to nx with p-value<0.001")

    def test_immutability_speedup(self):
        from pygrank.algorithms.pagerank import PageRank as Ranker
        from pygrank.algorithms.utils import preprocessor
        import scipy.stats
        nx_time = list()
        test_time = list()
        repeats = 50
        G = create_test_graph()
        ranker = Ranker(to_scipy=preprocessor('col'))
        tic = time.clock()
        for _ in range(repeats):
            ranker.rank(G)
        unhashed_time = time.clock()-tic
        ranker = Ranker(to_scipy=preprocessor('col', assume_immutability=True))
        tic = time.clock()
        for _ in range(repeats):
            ranker.rank(G)
        hashed_time = time.clock()-tic
        self.assertLessEqual(hashed_time, unhashed_time, msg="Hashing speedup")

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
        self.assertLess(pagerank['A']/sum(pagerank.values()), heatkernel['A']/sum(heatkernel.values()), msg="HeatKernel more local than PageRank")
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

    def test_link_auc(self):
        from pygrank.algorithms.pagerank import PageRank as Ranker
        from pygrank.metrics.multigroup import LinkAUC as LinkAUC
        G = create_test_graph()
        ranks1 = Ranker().rank(G, personalization = {"A": 1, "B": 1})
        ranks2 = Ranker().rank(G, personalization = {"F": 1, "G": 1})
        print('LinkAUC', LinkAUC(G, hops=2).evaluate({"group1": ranks1, "groups2": ranks2}))
        print('HopAUC', LinkAUC(G, hops=2).evaluate({"group1": ranks1, "groups2": ranks2}))

    def test_absorbing(self):
        from pygrank.algorithms.pagerank import AbsorbingRank as Ranker
        G = create_test_graph()
        ranks1 = Ranker(max_iters=1000).rank(G, personalization={"A": 1, "B": 1})
        ranks2 = Ranker(max_iters=1000).rank(G, personalization={"F": 1, "G": 1})
        from pygrank.metrics.multigroup import LinkAUC as LinkAUC
        print('Absorbing HopAUC', LinkAUC(G, hops=2).evaluate({"group1": ranks1, "groups2": ranks2}))


    def test_oversampling_top(self):
        from pygrank.algorithms.pagerank import AbsorbingRank as Ranker
        from pygrank.algorithms.oversampling import SeedOversampling
        G = create_test_graph()
        ranks1 = SeedOversampling(Ranker(max_iters=1000), method="top").rank(G, personalization={"A": 1, "B": 1})
        ranks2 = SeedOversampling(Ranker(max_iters=1000), method="top").rank(G, personalization={"F": 1, "G": 1})
        from pygrank.metrics.multigroup import LinkAUC as LinkAUC
        print('Top Oversampling + Absorbing HopAUC', LinkAUC(G, hops=2).evaluate({"group1": ranks1, "groups2": ranks2}))

    def test_venuerank(self):
        from pygrank.algorithms.pagerank import PageRank
        from pygrank.algorithms.postprocess import Ordinals
        G = nx.fast_gnp_random_graph(600, 0.001, seed=1)
        ranker1 = PageRank(alpha=0.9, max_iters=10000, converge_to_eigenvectors=True, tol=1.E-12)
        ranks1 = ranker1.rank(G, personalization={0: 1, 1: 1})
        ranker2 = PageRank(alpha=0.99, max_iters=10000, tol=1.E-12)
        ranks2 = ranker2.rank(G, personalization={0: 1, 1: 1})
        self.assertLess(ranker1.convergence.iteration, ranker2.convergence.iteration/10, msg="converge_to_eigenvectors (VenueRank) should be much faster in difficult-to-rank graphs")

        from scipy.stats import spearmanr
        corr = spearmanr(list(Ordinals().transform(ranks1).values()), list(Ordinals().transform(ranks2).values()))
        self.assertAlmostEqual(corr[0], 1., 4)

    def test_optimizer(self):
        from pygrank.algorithms.utils import optimize

        # a simple function
        p = optimize(loss=lambda p: (p[0]-2)**2+(p[1]-1)**4, max_vals=[5, 5], parameter_tol=1.E-8)
        self.assertAlmostEqual(p[0], 2, places=6, msg="Optimizer should easily optimize a convex function")
        self.assertAlmostEqual(p[1], 1, places=6, msg="Optimizer should easily optimize a convex function")

        # https://en.wikipedia.org/wiki/Test_functions_for_optimization

        # Beale function
        p = optimize(loss=lambda p: (1.5-p[0]+p[0]*p[1])**2+(2.25-p[0]+p[0]*p[1]**2)**2+(2.625-p[0]+p[0]*p[1]**3)**2, max_vals=[4.5, 4.5], min_vals=[-4.5, -4.5], parameter_tol=1.E-8)
        self.assertAlmostEqual(p[0], 3, places=6, msg="Optimizer should optimize the Beale function")
        self.assertAlmostEqual(p[1], 0.5, places=6, msg="Optimizer should optimize the Beale function")

        # Booth function
        p = optimize(loss=lambda p: (p[0]+2*p[1]-7)**2+(2*p[0]+p[1]-5)**2, max_vals=[10, 10], min_vals=[-10, -10], parameter_tol=1.E-8)
        self.assertAlmostEqual(p[0], 1, places=6, msg="Optimizer should optimize the Booth function")
        self.assertAlmostEqual(p[1], 3, places=6, msg="Optimizer should optimize the Booth function")

    def test_use_quotient_filter(self):
        from pygrank.algorithms.pagerank import PageRank
        from pygrank.algorithms.postprocess import Normalize

        G = create_test_graph()
        personalization = {"A": 1, "B": 1}

        ranks1 = PageRank(use_quotient=True).rank(G, personalization)
        ranks2 = PageRank(use_quotient=Normalize(method="sum")).rank(G, personalization)

        err = sum(abs(ranks1[v]-ranks2[v]) for v in G)
        self.assertAlmostEqual(err, 0, places=15, msg="use_quotient=Normalize(method='sum') should yield the same results (albeit a little slower than) True")


if __name__ == '__main__':
    unittest.main()
