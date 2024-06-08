from pygrank.core import backend
from pygrank.algorithms.filters.abstract import RecursiveGraphFilter


class SymmetricAbsorbingRandomWalks(RecursiveGraphFilter):
    """Implementation of partial absorbing random walks for *Lambda = (1-alpha)/alpha diag(absorption vector)*."""

    def __init__(self, alpha: float = 0.5, *args, **kwargs):
        """Initializes the symmetric random walk strategy for appropriate parameter values.

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
        self.absorption = (1 + (1 + 4 * self.degrees) ** 0.5) / 2
        self.personalization_skew = self.absorption / (self.absorption + self.degrees)
        self.sqrt_degrees = self.degrees / (self.absorption + self.degrees)
        self.sqrt_degrees_left = 1.0 / self.absorption

    def _end(self, *args, **kwargs):
        super()._end(*args, **kwargs)
        del self.absorption
        del self.degrees
        del self.sqrt_degrees
        del self.sqrt_degrees_left
        del self.personalization_skew

    def _formula(self, M, personalization, ranks, *args, **kwargs):
        return (
            backend.conv(ranks * self.sqrt_degrees_left, M) * self.sqrt_degrees
            + personalization * self.personalization_skew
        )

    def references(self):
        refs = super().references()
        refs[0] = (
            "symmetric partially absorbing random walks \\cite{krasanakis2022fast}"
        )
        return refs
