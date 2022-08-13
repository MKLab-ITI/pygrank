import pygrank as pg
import random
from tqdm import tqdm  # pip install tqdm (SimRank takes a while to conclude - this provides a progressbar
import numpy as np


class CoSimRank(pg.Postprocessor):
    def __init__(self, ranker=None, c=0.8):
        super().__init__(ranker)
        self.c = c
        self.known_ranks = dict()

    def _transform(self, ranks: pg.GraphSignal, **kwargs):
        if ranks.graph not in self.known_ranks:
            A = pg.preprocessor(normalization="col")(ranks.graph)
            S = np.eye(A.shape[0])
            self.known_ranks[ranks.graph] = [{v: repr for v, repr in zip(ranks.graph, pg.separate_cols(S))}]
            pow = 1
            for _ in range(100):
                S = S@A
                err = pow*np.sum(np.abs(S))/len(S)
                pow *= self.c
                self.known_ranks[ranks.graph].append({v: repr for v, repr in zip(ranks.graph, pg.separate_cols(S))})
                if err < 1.E-6:
                    break
        ret = 0
        for v in ranks:
            if ranks[v] != 0:
                pow = 1
                for known_ranks in self.known_ranks[ranks.graph]:
                    ret = ret + pow * known_ranks[v]*ranks[v]
                    pow *= self.c
        return pg.to_signal(ranks, ret)

    def _reference(self):
        return "SimRank"


def cos(a, b):
    return pg.dot(a,b)/(pg.dot(a, a)*pg.dot(b,b))**0.5


class CosRank(pg.Postprocessor):
    def __init__(self, ranker=None):
        super().__init__(ranker)
        self.known_ranks = dict()

    def _transform(self, ranks: pg.GraphSignal, **kwargs):
        if ranks.graph not in self.known_ranks:
            self.known_ranks[ranks.graph] = {v: self.ranker(pg.to_signal(ranks, {v: 1})) for v in tqdm(list(ranks.graph))}
        return {v: pg.dot(ranks, self.known_ranks[ranks.graph][v]) for v in ranks}


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
        result = (training >> algorithm)*(1-training)
        aucs.append(pg.AUC(test, exclude=training)(result))
        top = result >> pg.Top(10) >> pg.Threshold()
        prec = pg.PPV(test, exclude=training)(top)
        rec = pg.TPR(test, exclude=training)(top)
        ppvs.append(prec)
        tprs.append(rec)
        f1s.append(pg.safe_div(2*prec*rec, prec+rec))
        for neighbor in graph.neighbors(node):
            if not graph.has_edge(node, neighbor):
                graph.add_edge(node, neighbor)
        print(f"\r{algorithm.cite()}\t AUC {sum(aucs) / len(aucs):.3f}\t f1 {sum(f1s) / len(f1s):.3f}\t prec {sum(ppvs) / len(ppvs):.3f}\t rec {sum(tprs)/len(tprs):.3f}\t", end="")
    print()


graph = next(pg.load_datasets_graph(["citeseer"]))
evaluate(graph, pg.Tautology() >> CoSimRank())  # a variation of the very well-known SimRank implemented in this file
evaluate(graph, pg.PageRank())
evaluate(graph, pg.PageRank() >> CosRank())
evaluate(graph, pg.SymmetricAbsorbingRandomWalks())
