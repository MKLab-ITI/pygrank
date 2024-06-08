from pygrank.core import backend
from pygrank.algorithms.filters.abstract import RecursiveGraphFilter
from pygrank.core import to_signal, NodeRanking, preprocessor as default_preprocessor
from typing import Union, Optional
from pygrank.algorithms.filters.adhoc.pagerank.pagerank import PageRank


class LFPR(RecursiveGraphFilter):
    """Implements a locally fair variation of PageRank with universal fairness guarantees.
    Its preprocessor is overwritten to perform no renormalization and not to assume immutability,
    because it is a custom variation of column-based normalization that edits the adjacency matrix.
    """

    def __init__(
        self,
        alpha: float = 0.85,
        redistributor: Optional[Union[str, NodeRanking]] = None,
        target_prule: float = 1,
        fix_personalization: bool = True,
        *args,
        **kwargs,
    ):
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
        kwargs["preprocessor"] = default_preprocessor(
            assume_immutability=False, normalization=self.normalization
        )
        self.target_prule = target_prule
        self.redistributor = redistributor
        self.fix_personalization = fix_personalization
        super().__init__(*args, **kwargs)

    def normalization(self, M):
        import scipy.sparse

        sensitive = self.sensitive
        phi = self.phi
        outR = backend.conv(sensitive.np, M)
        outB = backend.conv(1.0 - sensitive.np, M)
        case1 = outR < (phi * (outR + outB))
        case2 = (1 - case1) * (outR != 0)
        case3 = (1 - case1) * (1 - case2)
        d = (
            case1 * backend.safe_inv(outB) * (1 - phi)
            + case2 * backend.safe_inv(outR) * phi
            + case3
        )
        Q = scipy.sparse.spdiags(d, 0, *M.shape)
        M = Q @ M
        self.outR = outR
        self.outB = outB
        return M

    def _prepare_graph(self, graph, personalization, sensitive, *args, **kwargs):
        personalization = to_signal(graph, personalization)
        sensitive = to_signal(personalization, sensitive)
        self.sensitive = sensitive
        self.phi = (
            backend.sum(sensitive) / backend.length(sensitive) * self.target_prule
        )
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

        dR = case1 * (phi - (1 - phi) * backend.safe_inv(outB) * outR) + case3 * phi
        dB = case2 * ((1 - phi) - phi * backend.safe_inv(outR) * outB) + case3 * (
            1 - phi
        )

        if self.redistributor is None or self.redistributor == "uniform":
            original_ranks = 1
        elif self.redistributor == "original":
            original_ranks = PageRank(
                alpha=self.alpha,
                tol=self.convergence.tol,
                preprocessor=default_preprocessor(
                    assume_immutability=False, normalization="col"
                ),
                convergence=self.convergence,
            )(personalization).np
        else:
            original_ranks = self.redistributor(personalization).np

        self.dR = dR
        self.dB = dB
        self.xR = backend.safe_div(
            original_ranks * sensitive.np, backend.sum(original_ranks * sensitive.np)
        )
        self.xB = backend.safe_div(
            original_ranks * (1 - sensitive.np),
            backend.sum(original_ranks * (1 - sensitive.np)),
        )
        super()._start(M, personalization, ranks, *args, **kwargs)

    def _formula(self, M, personalization, ranks, sensitive, *args, **kwargs):
        deltaR = backend.sum(
            ranks * self.dR
        )  # - self.personalization_residual_sensitive
        deltaB = backend.sum(
            ranks * self.dB
        )  # - self.personalization_residual_non_sensitive
        """ TODO: see if this is able to remove personalization removal from the end
        if deltaR < 0 or deltaB < 0:
            mm = backend.min(deltaR, deltaB)
            deltaR = deltaR - mm
            deltaB = deltaB - mm"""
        return (
            backend.conv(ranks, M) + deltaR * self.xR + deltaB * self.xB
        ) * self.alpha + personalization * (1 - self.alpha)

    def _end(self, M, personalization, ranks, *args, **kwargs):
        ranks.np = ranks.np - personalization.np * (1 - self.alpha)
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
        redistributor = "uniform" if self.redistributor is None else self.redistributor
        redistributor = (
            redistributor if isinstance(redistributor, str) else redistributor.cite()
        )
        refs.insert(2, f"{redistributor} rank redistribution strategy")
        return refs
