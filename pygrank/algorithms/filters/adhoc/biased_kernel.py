import warnings
from pygrank.core import backend
from pygrank.algorithms.filters.abstract import RecursiveGraphFilter


class BiasedKernel(RecursiveGraphFilter):
    """Heuristic kernel-like method that places emphasis on shorter random walks."""

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
        refs[0] = (
            "local ranking heuristic of low quality \\cite{krasanakis2020unsupervised}"
        )
        return refs
