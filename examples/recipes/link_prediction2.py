import pygrank as pg
import random
import numpy as np
import scipy.sparse
from typing import Optional


class CoSimRank(pg.Postprocessor):
    def __init__(
        self,
        ranker=None,
        weights=[0.5**i for i in range(3)],
        dims: int = 1024,
        sparsity: Optional[float] = None,
        beta: float = 1.0,
        normalization: str = "symmetric",
        assume_immutability: bool = True,
    ):
        super().__init__(pg.Tautology() if ranker is None else ranker)
        self.known_ranks = dict()
        self.embeddigns = dict()
        self.dims = dims
        self.weights = weights
        self.sparsity = sparsity
        self.beta = beta
        self.assume_immutability = assume_immutability
        self.normalization = normalization

    def _transform(self, ranks: pg.GraphSignal, **kwargs):
        if ranks.graph not in self.known_ranks or not self.assume_immutability:
            with pg.Backend("numpy"):
                A = pg.preprocessor(normalization=self.normalization)(ranks.graph)
                D = pg.degrees(pg.preprocessor(normalization="none")(ranks.graph))
                s = pg.sum(D) ** 0.5 / 2 if self.sparsity is None else self.sparsity
                D = (D / pg.max(D)) ** self.beta
                S = scipy.sparse.random(
                    self.dims,
                    A.shape[0],
                    density=1.0 / s,
                    data_rvs=lambda l: np.random.choice([-1, 1], size=l),
                    format="csc",
                )
                S = S @ scipy.sparse.spdiags(D, 0, *A.shape)
            self.embeddigns[ranks.graph] = pg.scipy_sparse_to_backend(S.T)
            self.known_ranks[ranks.graph] = (
                []
            )  # we know that the first term is zero and avoid direct embedding comparison
            for _ in range(len(self.weights)):
                S = S @ A
                self.known_ranks[ranks.graph].append(pg.scipy_sparse_to_backend(S))
        ret = 0
        on = pg.conv(ranks.np, self.embeddigns[ranks.graph])
        for weight, S in zip(self.weights, self.known_ranks[ranks.graph]):
            uv = pg.conv(on, S)
            ret = ret + weight * uv
        return pg.to_signal(ranks, ret)

    def _reference(self):
        return (
            "random graph embeddings \\cite{chen2019fast} with "
            + (
                "very sparse \\cite{li2006very}"
                if self.sparsity is None
                else str(int(self.sparse)) + "-sparse \\cite{achlioptas2003database}"
            )
            + " random projections"
        )


def evaluate(graph, algorithm):
    tprs = list()
    ppvs = list()
    f1s = list()
    aucs = list()
    for node in list(graph):
        neighbors = list(graph.neighbors(node))
        if len(neighbors) < 10:
            continue
        training = pg.to_signal(graph, {node: 1})
        test = pg.to_signal(graph, {neighbor: 1 for neighbor in neighbors})
        for neighbor in random.sample(neighbors, 1):
            assert graph.has_edge(node, neighbor)
            graph.remove_edge(node, neighbor)
            assert not graph.has_edge(node, neighbor)
            assert not graph.has_edge(neighbor, node)
        result = (training >> algorithm) * (1 - training)
        aucs.append(pg.AUC(test, exclude=training)(result))
        top = result >> pg.Top(10) >> pg.Threshold()
        prec = pg.PPV(test, exclude=training)(top)
        rec = pg.TPR(test, exclude=training)(top)
        ppvs.append(prec)
        tprs.append(rec)
        f1s.append(pg.safe_div(2 * prec * rec, prec + rec))
        for neighbor in graph.neighbors(node):
            if not graph.has_edge(node, neighbor):
                graph.add_edge(node, neighbor)
        print(
            f"\r{algorithm.cite()}\t AUC {sum(aucs) / len(aucs):.3f}\t f1 {sum(f1s) / len(f1s):.3f}\t prec {sum(ppvs) / len(ppvs):.3f}\t rec {sum(tprs)/len(tprs):.3f}\t",
            end="",
        )
    print()


graph = next(pg.load_datasets_graph(["citeseer"]))
evaluate(graph, CoSimRank())
evaluate(graph, pg.PageRank())
evaluate(graph, pg.SymmetricAbsorbingRandomWalks())
