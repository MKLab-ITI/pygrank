import warnings
from pygrank.core import backend
from pygrank.algorithms.filters.abstract_filters import RecursiveGraphFilter
from pygrank.algorithms.filters.low_pass import ClosedFormGraphFilter, LowPassRecursiveGraphFilter
from pygrank.core import to_signal, NodeRanking, preprocessor as default_preprocessor
from typing import Union, Optional


class PageRank(RecursiveGraphFilter):
    """A Personalized PageRank power method algorithm."""

    def __init__(self,
                 alpha: float = 0.85,
                 *args, **kwargs):
        """ Initializes the PageRank scheme parameters.
        Args:
            alpha: Optional. 1-alpha is the bias towards the personalization. Default alpha value is 0.85
                for historyical reasons. However, in large graphs it is often preferred to set this
                argument to 0.9.
        Example:
            >>> import pygrank as pg
            >>> algorithm = pg.PageRank(alpha=0.99, tol=1.E-9) # tol passed to the ConvergenceManager
            >>> graph, seed_nodes = ...
            >>> ranks = algorithm(graph, {v: 1 for v in seed_nodes})
        """
        self.alpha = alpha
        super().__init__(*args, **kwargs)

    def _start(self, M, personalization, ranks, *args, **kwargs):
        #self.dangling_weights = backend.degrees(M)
        #self.is_dangling = self.dangling_weights/backend.sum(self.dangling_weights)
        super()._start(M, personalization, ranks, *args, **kwargs)

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        # TODO: return self.alpha * (ranks * M + backend.sum(ranks[self.is_dangling]) * personalization) + (1 - self.alpha) * personalization
        return backend.conv(ranks, M) * self.alpha + personalization * (1 - self.alpha)

    def _end(self, M, personalization, ranks, *args, **kwargs):
        #del self.is_dangling
        super()._end(M, personalization, ranks, *args, **kwargs)

    def references(self):
        refs = super().references()
        refs[0] = "personalized PageRank \\cite{page1999pagerank}"
        refs.insert(1, f"diffusion rate {self.alpha:.3f}")
        return refs


class DijkstraRank(RecursiveGraphFilter):
    """A ranking algorithm that assigns node ranks loosely increasing with the minimum distance from a seed."""
    def __init__(self, degradation=0.1, *args, **kwargs):
        self.degradation = degradation
        super().__init__(*args, **kwargs)

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        prev_ranks = ranks
        ranks = backend.conv(ranks, M)*self.degradation
        choice = backend.cast(ranks.np > prev_ranks.np)
        ranks.np = choice*ranks.np + (1-choice)*prev_ranks.np
        return ranks


class PageRankClosed(ClosedFormGraphFilter):
    """ PageRank closed filter."""

    def __init__(self,
                 alpha: float = 0.85,
                 *args, **kwargs):
        """ Initializes the PageRank scheme parameters.
        Args:
            alpha: Optional. 1-alpha is the bias towards the personalization. Default alpha value is 0.85
                for historyical reasons. However, in large graphs it is often preferred to set this
                argument to 0.9.
        Example:
            >>> import pygrank as pg
            >>> algorithm = pg.PageRankClosed(alpha=0.99, tol=1.E-9) # tol passed to the ConvergenceManager
            >>> graph, seed_nodes = ...
            >>> ranks = algorithm(graph, {v: 1 for v in seed_nodes})
        """
        self.alpha = alpha
        super().__init__(*args, **kwargs)

    def _coefficient(self, previous_coefficient):
        return 1. if previous_coefficient is None else (previous_coefficient * self.alpha)

    def references(self):
        refs = super().references()
        refs[0] = "polynomial personalized PageRank \\cite{page1999pagerank}"
        refs.insert(1, f"diffusion rate {self.alpha:.3f}")
        return refs


class HeatKernel(ClosedFormGraphFilter):
    """ Heat kernel filter."""

    def __init__(self,
                 t: float = 3,
                 *args, **kwargs):
        """ Initializes filter parameters.

        Args:
            t: Optional. How many hops until the importance of new nodes starts decreasing. Default value is 5.

        Example:
            >>> from pygrank.algorithms import HeatKernel
            >>> algorithm = HeatKernel(t=3, tol=1.E-9) # tol passed to the ConvergenceManager
            >>> graph, seed_nodes = ...
            >>> ranks = algorithm(graph, {v: 1 for v in seed_nodes})
        """
        self.t = t
        super().__init__(*args, **kwargs)

    def _coefficient(self, previous_coefficient):
        # backend.exp(-self.t)
        return 1. if previous_coefficient is None else (
                previous_coefficient * self.t / (self.convergence.iteration + 1))

    def references(self):
        refs = super().references()
        refs[0] = "HeatKernel \\cite{chung2007heat}"
        refs.insert(1, f"emphasis on {self.t}-hop distances")
        return refs


class AbsorbingWalks(RecursiveGraphFilter):
    """ Implementation of partial absorbing random walks for Lambda = (1-alpha)/alpha diag(absorption vector).
    To determine parameters based on symmetricity principles, use *SymmetricAbsorbingRandomWalks*."""

    def __init__(self,
                 alpha: float = 1 - 1.E-6,
                 *args, **kwargs):
        """ Initializes filter parameters. The filter can model PageRank for appropriate parameter values,
        but is in principle a generalization that allows custom absorption rates per node (when not given, these are I).

        Args:
            alpha: Optional. (1-alpha)/alpha is the absorption rate of the random walk multiplied with individual node
                absorption rates. This is chosen to yield the
                same underlying meaning as PageRank (for which Lambda = alpha Diag(degrees) ) when the same parameter
                value alpha is chosen. Default is 1-1.E-6 per the respective publication.

        Example:
            >>> from pygrank.algorithms import AbsorbingWalks
            >>> algorithm = AbsorbingWalks(1-1.E-6, tol=1.E-9) # tol passed to the ConvergenceManager
            >>> graph, seed_nodes = ...
            >>> ranks = algorithm(graph, {v: 1 for v in seed_nodes})

        Example (same outcome, explicit absorption rate definition):
            >>> from pygrank.algorithms import AbsorbingWalks
            >>> algorithm = AbsorbingWalks(1-1.E-6, tol=1.E-9) # tol passed to the ConvergenceManager
            >>> graph, seed_nodes = ...
            >>> ranks = algorithm(graph, {v: 1 for v in seed_nodes}, absorption={v: 1 for v in graph})
        """

        super().__init__(*args, **kwargs)
        self.alpha = alpha  # typecast to make sure that a graph is not accidentally the first argument

    def _start(self, M, personalization, ranks, absorption=None, **kwargs):
        self.absorption = to_signal(personalization.graph, absorption) * ((1 - self.alpha) / self.alpha)
        self.degrees = backend.degrees(M)

    def _end(self, *args, **kwargs):
        super()._end(*args, **kwargs)
        del self.absorption
        del self.degrees

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        ret = (backend.conv(ranks, M) * self.degrees + personalization * self.absorption) / (
                self.absorption + self.degrees)
        return ret

    def references(self):
        refs = super().references()
        refs[0] = "partially absorbing random walks \\cite{wu2012learning}"
        return refs


class BiasedKernel(RecursiveGraphFilter):
    """ Heuristic kernel-like method that places emphasis on shorter random walks."""

    def __init__(self, alpha: float = 0.85, t: float = 1, *args, **kwargs):
        self.alpha = alpha
        self.t = t
        super().__init__(*args, **kwargs)
        warnings.warn("BiasedKernel is a low-quality heuristic", stacklevel=2)

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        a = self.alpha * self.t / self.convergence.iteration
        return personalization + a * (backend.conv(ranks, M) * personalization) - ranks

    def references(self):
        refs = super().references()
        refs[0] = "local ranking heuristic of low quality \\cite{krasanakis2020unsupervised}"
        return refs


class LFPR(RecursiveGraphFilter):
    """Implements a locally fair variation of PageRank with universal fairness guarantees.
    Its preprocessor is overwritten to perform no renormalization and not to assume immutability,
    because it is a custom variation of column-based normalization that edits the adjacency matrix.
    """

    def __init__(self,
                 alpha: float = 0.85,
                 redistributor: Optional[Union[str, NodeRanking]] = None,
                 target_prule: float = 1,
                 fix_personalization: bool = True,
                 *args, **kwargs):
        """
        Initializes the locally fair random walk filter's parameters.
        Args:
            alpha: Corresponds to the respective parameter of PageRank.
            redistributor: Redistribution strategy. If None (default) a uniform redistribution is
                performed. If "original", a PageRank algorithm with colum-based normalization is run and used.
                Otherwise, it can be a node ranking algorithm that estimates how much importance to
                place on each node when redistributing non-fair random walk probability remainders.
            target_prule: Target pRule value to achieve. Default is 1.
        """
        self.alpha = alpha

        # TODO: find a way to assume immutability with transparency
        kwargs["preprocessor"] = default_preprocessor(assume_immutability=False, normalization=self.normalization)
        self.target_prule = target_prule
        self.redistributor = redistributor
        self.fix_personalization = fix_personalization
        super().__init__(*args, **kwargs)

    def normalization(self, M):
        import scipy.sparse
        sensitive = self.sensitive
        phi = self.phi
        outR = backend.conv(sensitive.np, M)
        outB = backend.conv(1. - sensitive.np, M)
        case1 = (outR < (phi * (outR + outB)))
        case2 = ((1 - case1) * (outR != 0))
        case3 = ((1 - case1) * (1 - case2))
        d = case1*backend.safe_inv(outB)*(1-phi) + case2*backend.safe_inv(outR)*phi + case3
        Q = scipy.sparse.spdiags(d, 0, *M.shape)
        M = Q @ M
        self.outR = outR
        self.outB = outB
        return M

    def _prepare_graph(self, graph, personalization, sensitive, *args, **kwargs):
        personalization = to_signal(graph, personalization)
        sensitive = to_signal(personalization, sensitive)
        self.sensitive = sensitive
        self.phi = backend.sum(sensitive) / backend.length(sensitive) * self.target_prule
        """if self.fix_personalization:
            self.personalization_residual_sensitive = backend.sum(personalization*sensitive)
            self.personalization_residual_non_sensitive = backend.sum(personalization*sensitive)
        else:
            self.personalization_residual_sensitive = 0
            self.personalization_residual_non_sensitive = 0"""
        return graph

    def _start(self, M, personalization, ranks, sensitive, *args, **kwargs):
        sensitive = to_signal(ranks, sensitive)
        outR = self.outR  # backend.conv(sensitive.np, M)
        outB = self.outB  # backend.conv(1.-sensitive.np, M)
        phi = backend.sum(sensitive) / backend.length(sensitive) * self.target_prule
        case1 = outR < (phi * (outR + outB))
        case2 = (1 - case1) * (outR != 0)
        case3 = (1 - case1) * (1 - case2)

        dR = case1*(phi - (1 - phi) * backend.safe_inv(outB) * outR) + case3*phi
        dB = case2*((1 - phi) - phi * backend.safe_inv(outR) * outB) + case3*(1-phi)

        if self.redistributor is None or self.redistributor == "uniform":
            original_ranks = 1
        elif self.redistributor == "original":
            original_ranks = PageRank(alpha=self.alpha,
                                      tol=self.convergence.tol,
                                      preprocessor=default_preprocessor(assume_immutability=False, normalization="col"),
                                      convergence=self.convergence)(personalization).np
        else:
            original_ranks = self.redistributor(personalization).np

        self.dR = dR
        self.dB = dB
        self.xR = backend.safe_div(original_ranks * sensitive.np, backend.sum(original_ranks * sensitive.np))
        self.xB = backend.safe_div(original_ranks * (1 - sensitive.np), backend.sum(original_ranks * (1 - sensitive.np)))
        super()._start(M, personalization, ranks, *args, **kwargs)

    def _formula(self, M, personalization, ranks, sensitive, *args, **kwargs):
        deltaR = backend.sum(ranks * self.dR)# - self.personalization_residual_sensitive
        deltaB = backend.sum(ranks * self.dB)# - self.personalization_residual_non_sensitive
        """ TODO: see if this is able to remove personalization removal from the end
        if deltaR < 0 or deltaB < 0:
            mm = backend.min(deltaR, deltaB)
            deltaR = deltaR - mm
            deltaB = deltaB - mm"""
        return (backend.conv(ranks, M) + deltaR * self.xR + deltaB * self.xB) * self.alpha + personalization * (
                1 - self.alpha)

    def _end(self, M, personalization, ranks, *args, **kwargs):
        ranks.np = ranks.np - personalization.np*(1-self.alpha)
        del self.xR
        del self.xB
        del self.dR
        del self.dB
        del self.sensitive
        del self.phi
        del self.outR
        del self.outB
        super()._end(M, personalization, ranks, *args, **kwargs)

    def references(self):
        refs = super().references()
        refs[0] = "fairness-aware PageRank \\cite{tsioutsiouliklis2021fairness}"
        refs.insert(1, f"diffusion rate {self.alpha:.3f}")
        redistributor = 'uniform' if self.redistributor is None else self.redistributor
        redistributor = redistributor if isinstance(redistributor, str) else redistributor.cite()
        refs.insert(2, f"{redistributor} rank redistribution strategy")
        return refs


class SymmetricAbsorbingRandomWalks(RecursiveGraphFilter):
    """ Implementation of partial absorbing random walks for *Lambda = (1-alpha)/alpha diag(absorption vector)*."""

    def __init__(self,
                 alpha: float = 0.5,
                 *args, **kwargs):
        """ Initializes the symmetric random walk strategy for appropriate parameter values.

        Args:
            alpha: Optional. (1-alpha)/alpha is the absorption rate of the random walk multiplied with individual node
                absorption rates. This is chosen to yield the
                same underlying meaning as PageRank (for which Lambda = alpha Diag(degrees) ) when the same parameter
                value alpha is chosen. Default is 0.5 to match the approach of [krasanakis2022fast],
                which uses absorption rate 1. Ideally, to set this parameter, refer to *AbsorbingWalks*.

        Example:
            >>> from pygrank.algorithms import AbsorbingWalks
            >>> algorithm = AbsorbingWalks(1-1.E-6, tol=1.E-9)
            >>> graph, seed_nodes = ...
            >>> ranks = algorithm(graph, {v: 1 for v in seed_nodes})

        Example (same outcome, explicit absorption rate definition):
            >>> from pygrank.algorithms import AbsorbingWalks
            >>> algorithm = AbsorbingWalks(1-1.E-6, tol=1.E-9)
            >>> graph, seed_nodes = ...
            >>> ranks = algorithm(graph, {v: 1 for v in seed_nodes}, absorption={v: 1 for v in graph})
        """

        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def _start(self, M, personalization, ranks, **kwargs):
        self.degrees = backend.degrees(M)
        self.absorption = (1+(1+4*self.degrees)**0.5)/2
        self.personalization_skew = self.absorption / (self.absorption + self.degrees)
        self.sqrt_degrees = (self.degrees / (self.absorption + self.degrees))
        self.sqrt_degrees_left = 1./self.absorption

    def _end(self, *args, **kwargs):
        super()._end(*args, **kwargs)
        del self.absorption
        del self.degrees
        del self.sqrt_degrees
        del self.sqrt_degrees_left
        del self.personalization_skew

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        return backend.conv(ranks*self.sqrt_degrees_left, M) * self.sqrt_degrees + personalization * self.personalization_skew

    def references(self):
        refs = super().references()
        refs[0] = "symmetric partially absorbing random walks \\cite{krasanakis2022fast}"
        return refs
