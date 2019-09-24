import warnings


class Conductance:
    def __init__(self, G, max_rank=1):
        self.G = G
        self.max_rank = max_rank

    def evaluate(self, ranks):
        external_edges = sum(ranks.get(i, 0)*(self.max_rank-ranks.get(j, 0)) for i, j in self.G.edges())
        internal_edges = sum(ranks.get(i, 0)*ranks.get(j, 0) for i, j in self.G.edges())
        if internal_edges > self.G.number_of_edges()/2:
            internal_edges = self.G.number_of_edges()-internal_edges # user the smallest partition as reference
        if not self.G.is_directed():
            external_edges += sum(ranks.get(j, 0) * (self.max_rank - ranks.get(i, 0)) for i, j in self.G.edges())
            internal_edges *= 2
        if internal_edges == 0:
            return float('inf')
        return external_edges / internal_edges


class Density:
    def __init__(self, G):
        self.G = G

    def evaluate(self, ranks):
        internal_edges = sum(ranks.get(i, 0) * ranks.get(j, 0) for i,j  in self.G.edges())
        expected_edges = sum(ranks.values()) ** 2 - sum(rank ** 2 for rank in ranks.values()) # without self-loops
        if internal_edges == 0:
            return 0
        return internal_edges / expected_edges


class FastSweep:
    def __init__(self, G, base_metric=None):
        self.G = G
        self.base_metric = Conductance(G) if base_metric is None else base_metric
        warnings.warn("FastSweep is still under development (its implementation may be incorrect)", stacklevel=2)

    def evaluate(self, ranks):
        # TODO: check implementation
        ranks = {v: ranks[v] / self.G.degree(v) for v in ranks}
        max_diff = 0
        max_diff_val = 0
        prev_rank = 0
        for v in sorted(ranks, key=ranks.get, reverse=True):
            if prev_rank > 0:
                diff = (prev_rank - ranks[v]) / prev_rank
                if diff > max_diff:
                    max_diff = diff
                    max_diff_val = ranks[v]
            prev_rank = ranks[v]
        return self.base_metric.evaluate({v: 1 for v in ranks.keys() if ranks[v] >= max_diff_val})
