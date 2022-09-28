from pygrank.fastgraph.fastgraph import Graph


class AdjacencyWrapper(Graph):
    def __init__(self, adj, directed=True):
        self.adj = adj
        self.num_nodes = adj.shape[0]
        self.directed = directed

    def is_directed(self):
        return self.directed

    def __iter__(self):
        return range(self.num_nodes).__iter__()

    def __len__(self):
        return self.num_nodes

    def to_scipy_sparse_array(self):
        return self.adj
