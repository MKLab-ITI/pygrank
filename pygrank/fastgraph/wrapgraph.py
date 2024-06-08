from pygrank.fastgraph.fastgraph import Graph
from pygrank.core import backend


class AdjacencyWrapper(Graph):
    def __init__(self, adj, directed=True):
        if hasattr(adj, "array"):
            adj = adj.array
        if hasattr(adj, "tocsr"):  # recognize scipy
            adj = adj.tocsr()
        else:  # TODO: try to recover "numpy" from cors
            raise Exception(
                "Wrapping adjacency matrices is only possible for scipy spars matrices or the outcomes of preprocessors with cors=True"
            )
        """elif hasattr(adj, "__pygrank_preprocessed"):
                    if "numpy" not in adj.__pygrank_preprocessed:
                        raise Exception("To create an adjacency wrapper from an adjacency that was preprocessed in a backend not directly compatible with numpy, enable cors=True to the preprocessor that used it")
                    adj = adj.__pygrank_preprocessed["numpy"]
                """
        self.adj = adj
        # self.num_nodes = backend.shape0(adj)
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
