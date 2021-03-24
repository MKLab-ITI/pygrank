import scipy
import numpy as np
import warnings
import pygrank.algorithms.utils
from pygrank.algorithms.utils import _call, _ensure_all_used



class PageRank:
    """A Personalized PageRank power method algorithm. Supports warm start."""

    def __init__(self, alpha=0.85, to_scipy=None, to_numpy=None, convergence=None, use_quotient=True, converge_to_eigenvectors=False, **kwargs):
        """ Initializes the PageRank scheme parameters.

        Attributes:
            alpha: Optional. 1-alpha is the bias towards the personalization. Default value is 0.85.
            to_scipy: Optional. Method to extract a scipy sparse matrix from a networkx graph.
                If None (default), pygrank.algorithms.utils.preprocessor is used with the keyword arguments is
                used by this constructor.
            to_numpy: Optional. Method to extract a personalization from a dictionary of graph nodes.
                If None (default), pygrank.algorithms.utils.vectorize is used with the keyword arguments is
                used by this constructor.
            convergence: Optional. The ConvergenceManager that determines when iterations stop. If None (default),
                a ConvergenceManager with the keyword arguments given to this constructor is created.
            use_quotient: Optional. If True (default) performs a L1 re-normalization of ranks after each iteration.
                This significantly speeds ups the convergence speed of symmetric normalization (col normalization
                preserves the L1 norm during computations on its own). Can also pass a pygrank.algorithm.postprocess
                rank editing mechanism to perform any kind of normalization through its postprocess method. This slows
                down computations due to needing to convert ranks between scipy and dicts after each iteration.
                Can pass False or None to ignore this parameter's functionality.
            converge_to_eigenvectors: Optional. If True (default is False) the outcome of ranking does not depend on
                the alpha parameters and only weakly on potential personalization. Instead ranking strongly biased
                towards the principal eigenvector. If more than one near-max eigenvalues of to_scipy(graph) exist
                then this scheme selects one based on the personalization scheme, otherwise the principal eigenvector
                is outputted. When the graph is bipartite, using this and the argument use_quotient=False
                effectively toggles the behavior described in
                `VenueRank: Identifying Venues that Contribute to Artist Popularity`.

        Example:
            >>> from pygrank.algorithms import pagerank
            >>> algorithm = pagerank.PageRank(alpha=0.99, tol=1.E-9) # tol passed to the ConvergenceManager
        """
        self.alpha = float(alpha) # typecast to make sure that a graph is not accidentally the first argument
        self.to_scipy = _call(pygrank.algorithms.utils.preprocessor, kwargs) if to_scipy is None else to_scipy
        self.to_numpy = _call(pygrank.algorithms.utils.vectorize, kwargs) if to_numpy is None else to_numpy
        self.convergence = _call(pygrank.algorithms.utils.ConvergenceManager, kwargs) if convergence is None else convergence
        _ensure_all_used(kwargs, [pygrank.algorithms.utils.preprocessor, pygrank.algorithms.utils.vectorize, pygrank.algorithms.utils.ConvergenceManager])
        self.use_quotient = None if use_quotient == False else use_quotient
        self.converge_to_eigenvectors = converge_to_eigenvectors

    def rank(self, G, personalization=None, warm_start=None, as_dict=True, *args, **kwargs):
        M = self.to_scipy(G)
        personalization = self.to_numpy(G, personalization)
        ranks = personalization if warm_start is None else self.to_numpy(G, warm_start)
        is_dangling = np.where(np.array(M.sum(axis=1)).flatten() == 0)[0]
        self.convergence.start()
        while not self.convergence.has_converged(ranks):
            ranks = self.alpha * (ranks * M + np.sum(ranks[is_dangling]) * personalization) + (1 - self.alpha) * personalization
            #ranks = self.alpha * (ranks * M) + (1 - self.alpha) * personalization
            if self.use_quotient == True:
                ranks = ranks / ranks.sum()
            elif self.use_quotient is not None:
                ranks = dict(zip(G.nodes(), map(float, ranks)))
                ranks = self.use_quotient.transform(ranks, *args, **kwargs)
                ranks = np.array([ranks.get(n, 0) for n in G], dtype=float)
            if self.converge_to_eigenvectors:
                personalization = ranks
        return dict(zip(G.nodes(), map(float, ranks))) if as_dict else ranks


class HeatKernel:
    """ Heat kernel filter."""

    def __init__(self, t=3, to_scipy=None, to_numpy=None, convergence=None, **kwargs):
        """ Initializes the HearKernel filter parameters.

        Attributes:
            t: Optional. How many hops until the importance of new nodes starts decreasing. Default value is 5.
            to_scipy: Optional. Method to extract a scipy sparse matrix from a networkx graph.
                If None (default), pygrank.algorithms.utils.preprocessor is used with the keyword arguments is
                used by this constructor.
            to_numpy: Optional. Method to extract a personalization from a dictionary of graph nodes.
                If None (default), pygrank.algorithms.utils.vectorize is used with the keyword arguments is
                used by this constructor.
            convergence: Optional. The ConvergenceManager that determines when iterations stop. If None (default),
                a ConvergenceManager with the keyword arguments given to this constructor is created.

        Example:
            >>> from pygrank.algorithms import pagerank
            >>> algorithm = pagerank.HeatKernel(t=5, tol=1.E-9) # tol passed to the ConvergenceManager
        """
        self.t = t
        self.to_scipy = _call(pygrank.algorithms.utils.preprocessor, kwargs) if to_scipy is None else to_scipy
        self.convergence = _call(pygrank.algorithms.utils.ConvergenceManager, kwargs) if convergence is None else convergence
        self.to_numpy = _call(pygrank.algorithms.utils.vectorize, kwargs) if to_numpy is None else to_numpy
        _ensure_all_used(kwargs, [pygrank.algorithms.utils.preprocessor, pygrank.algorithms.utils.vectorize, pygrank.algorithms.utils.ConvergenceManager])

    def rank(self, G, personalization=None, *args, as_dict=True, **kwargs):
        M = self.to_scipy(G)
        personalization = self.to_numpy(G, personalization)
        coefficient = np.exp(-self.t)
        ranks = personalization*coefficient
        self.convergence.start()
        Mpower = M
        while not self.convergence.has_converged(ranks):
            coefficient *= self.t/(self.convergence.iteration+1)
            Mpower *= M
            ranks += personalization*Mpower*coefficient
        ranks = ranks/ranks.sum()
        return dict(zip(G.nodes(), map(float, ranks))) if as_dict else ranks


class AbsorbingRank:
    """ Implementation of partial absorbing random walks for Lambda = diag(absorbtion vector), e.g. Lambda = aI
    Wu, Xiao-Ming, et al. "Learning with partially absorbing random walks." Advances in neural information processing systems. 2012.
    """

    def __init__(self, alpha=1-1.E-6, to_scipy=None, to_numpy=None, use_quotient=True, convergence=None, **kwargs):
        """ Initializes the AbsorbingRank filter parameters.

        Attributes:
            alpha: Optional. (1-alpha)/alpha is the absorbsion rate of the random walk. This is chosen to yield the
                same underlying meaning as PageRank (for which Lambda = a Diag(degrees) )
            to_scipy: Optional. Method to extract a scipy sparse matrix from a networkx graph.
                If None (default), pygrank.algorithms.utils.preprocessor is used with the keyword arguments is
                used by this constructor.
            to_numpy: Optional. Method to extract a personalization from a dictionary of graph nodes.
                If None (default), pygrank.algorithms.utils.vectorize is used with the keyword arguments is
                used by this constructor.
            convergence: Optional. The ConvergenceManager that determines when iterations stop. If None (default),
                a ConvergenceManager with the keyword arguments given to this constructor is created.
            use_quotient: Optional. If True (default) performs a L1 re-normalization of ranks after each iteration.
                This significantly speeds ups the convergence speed of symmetric normalization (col normalization
                preserves the L1 norm during computations on its own). Can also pass a pygrank.algorithm.postprocess
                filter to perform any kind of normalization through its postprocess method. Note that these can slow
                down computations due to needing to convert ranks between skipy and maps after each iteration.
                Can pass False or None to ignore this parameter's functionality.

        Example:
            >>> from pygrank.algorithms import pagerank
            >>> algorithm = pagerank.HeatKernel(t=5, tol=1.E-9) # tol passed to the ConvergenceManager
        """
        self.alpha = float(alpha) # typecast to make sure that a graph is not accidentally the first argument
        self.to_scipy = _call(pygrank.algorithms.utils.preprocessor, kwargs) if to_scipy is None else to_scipy
        self.convergence = _call(pygrank.algorithms.utils.ConvergenceManager, kwargs) if convergence is None else convergence
        self.to_numpy = _call(pygrank.algorithms.utils.vectorize, kwargs) if to_numpy is None else to_numpy
        _ensure_all_used(kwargs, [pygrank.algorithms.utils.preprocessor, pygrank.algorithms.utils.vectorize, pygrank.algorithms.utils.ConvergenceManager])
        self.use_quotient = None if use_quotient == False else use_quotient

    def rank(self, G, personalization=None, attraction=None, absorption=None, warm_start=None, residuals=None, *args, as_dict=False, **kwargs):
        M = self.to_scipy(G)
        degrees = np.array(M.sum(axis=1)).flatten()
        personalization = self.to_numpy(G, personalization)
        ranks = personalization if warm_start is None else self.to_numpy(G, warm_start)

        #is_dangling = np.where(degrees == 0)[0]
        self.convergence.start()
        attract = np.repeat(1.0, len(G)) if attraction is None else self.to_numpy(G, attraction)
        diag_of_lamda = (1-self.alpha)/self.alpha * (np.repeat(1.0, len(G)) if absorption is None else np.array([absorption.get(n, 0) for n in G], dtype=float))

        if residuals is not None:
            residuals = [( np.array([residual.get(n, 0) for n in G], dtype=float),
                           np.array([1 if residual.get(n, 0)!=0 else 0 for n in G], dtype=float) )
                          for residual in residuals]

        while not self.convergence.has_converged(ranks):
            Mfair = M
            if residuals is not None:
                for residual, mask in residuals:
                    masked_ranks = mask*ranks
                    masked_rank_sum = masked_ranks.sum()
                    if masked_rank_sum != 0:
                        masked_ranks /= masked_ranks.sum()
                    Mfair += np.cross(residual, masked_ranks)
            #ranks = (ranks * attract * Mfair + sum(ranks[is_dangling]) * personalization)*degrees/(diag_of_lamda+degrees) + personalization*diag_of_lamda/(diag_of_lamda+degrees)
            ranks = (ranks * attract * Mfair) * degrees / (diag_of_lamda + degrees) + personalization * diag_of_lamda / (diag_of_lamda + degrees)
            if self.use_quotient == True:
                ranks = ranks/ranks.sum()
            elif self.use_quotient is not None:
                ranks = dict(zip(G.nodes(), map(float, ranks)))
                ranks = self.use_quotient.transform(ranks, *args, **kwargs)
                ranks = np.array([ranks.get(n, 0) for n in G], dtype=float)
        return dict(zip(G.nodes(), map(float, ranks))) if as_dict else ranks


class BiasedKernel:
    """ Heuristic kernel-like method that places emphasis on shorter random walks."""

    def __init__(self, alpha=0.85, t=1, to_scipy=None, convergence=None, **kwargs):
        self.alpha = alpha
        self.t = t
        self.to_scipy = _call(pygrank.algorithms.utils.preprocessor, kwargs) if to_scipy is None else to_scipy
        self.convergence = _call(pygrank.algorithms.utils.ConvergenceManager, kwargs) if convergence is None else convergence
        _ensure_all_used(kwargs, [pygrank.algorithms.utils.preprocessor, pygrank.algorithms.utils.ConvergenceManager])
        warnings.warn("BiasedKernel is a low-quality heuristic", stacklevel=2)

    def rank(self, G, personalization=None, warm_start=None):
        M = self.to_scipy(G)
        degrees = np.array(M.sum(axis=1)).flatten()

        personalization = np.repeat(1.0, len(G)) if personalization is None else np.array([personalization.get(n, 0) for n in G], dtype=float)
        if personalization.sum() == 0:
            raise Exception("The personalization vector should contain at least one non-zero entity")
        personalization = personalization / personalization.sum()
        ranks = personalization if warm_start is None else np.array([warm_start.get(n, 0) for n in G], dtype=float)

        is_dangling = scipy.where(degrees == 0)[0]
        self.convergence.start()
        while not self.convergence.has_converged(ranks):
            a = self.alpha*self.t/self.convergence.iteration
            ranks = personalization + a * ((ranks * M + sum(ranks[is_dangling]) * personalization) - ranks)
            ranks = ranks/ranks.sum()

        ranks = dict(zip(G.nodes(), map(float, ranks)))
        return ranks
