import scipy
import algorithms.utils


class PageRank:
    def __init__(self, alpha=0.85, normalization='auto', convergence_manager=None):
        self.alpha = alpha
        self.normalization = normalization
        self.convergence = algorithms.utils.ConvergenceManager() if convergence_manager is None else convergence_manager

    def rank(self, G, personalization=None, warm_start=None):
        M = algorithms.utils.to_scipy_sparse_matrix(G, self.normalization)
        degrees = scipy.array(M.sum(axis=1)).flatten()

        personalization = scipy.repeat(1.0, len(G)) if personalization is None else scipy.array([personalization.get(n, 0) for n in G], dtype=float)
        personalization = personalization / personalization.sum()
        ranks = personalization if warm_start is None else scipy.array([warm_start.get(n, 0) for n in G], dtype=float)

        is_dangling = scipy.where(degrees == 0)[0]
        self.convergence.reset()
        while not self.convergence.has_converged(ranks):
            ranks = self.alpha * (ranks * M + sum(ranks[is_dangling]) * personalization) + (1 - self.alpha) * personalization
            ranks = ranks/ranks.sum()

        ranks = dict(zip(G.nodes(), map(float, ranks)))
        return ranks
