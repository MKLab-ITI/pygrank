from pygrank.algorithms.filters.abstract.filter import GraphFilter
from pygrank.core import (
    GraphSignal,
    BackendGraph,
    BackendPrimitive,
)
from pygrank.core import backend
from pygrank.algorithms.postprocess import Postprocessor
from typing import Union


class RecursiveGraphFilter(GraphFilter):
    """Implements a graph filter described through a recursion ranks = formula(G, ranks)"""

    def __init__(
        self,
        use_quotient: Union[bool, Postprocessor] = True,
        converge_to_eigenvectors: bool = False,
        *args,
        **kwargs
    ):
        """
        Args:
            use_quotient: Optional. If True (default) performs a L1 re-normalization of ranks after each iteration.
                This significantly speeds up the convergence speed of symmetric normalization (col normalization
                preserves the L1 norm during computations on its own). Provide `pygrank.Postprocessor` or other
                callable instances to adjust node scores after each iteration.
                Can pass False or None to ignore this functionality and make recursive filter outcome equal to
                its expansion.
        """
        super().__init__(*args, **kwargs)
        self.use_quotient = use_quotient
        self.converge_to_eigenvectors = converge_to_eigenvectors

    def _step(self, M, personalization, ranks, *args, **kwargs):
        ranks.np = self._formula(M, personalization, ranks, *args, **kwargs)
        if isinstance(ranks.np, GraphSignal):
            ranks.np = ranks.np.np

        if isinstance(self.use_quotient, Postprocessor):
            ranks.np = self.use_quotient(ranks)
        elif self.use_quotient:
            ranks.np = backend.safe_div(ranks, backend.sum(ranks))
        if self.converge_to_eigenvectors:
            personalization.np = ranks.np

    def _formula(
        self,
        M: BackendGraph,
        personalization: BackendPrimitive,
        ranks: BackendPrimitive,
        *args,
        **kwargs
    ):
        raise Exception(
            "Use a derived class of RecursiveGraphFilter that implements the _formula method"
        )

    def references(self):
        refs = super().references()
        if self.converge_to_eigenvectors:
            refs += ["unbiased eigenvector convergence \\cite{krasanakis2018venuerank}"]
        return refs
