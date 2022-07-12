from scipy.sparse import coo_matrix


class Graph:
    def __init__(self, directed=False):
        self.edge_row = list()
        self.edge_col = list()
        self.node_map = dict()
        self.directed = directed

    def add_node(self, node):
        if node not in self.node_map:
            self.node_map[node] = len(self.node_map)
        return self.node_map[node]

    def add_edge(self, u, v):
        u = self.add_node(u)
        v = self.add_node(v)
        self.edge_row.append(u)
        self.edge_col.append(v)
        if not self.directed:
            self.edge_row.append(v)
            self.edge_col.append(u)

    def is_directed(self):
        return self.directed

    def nodes(self):
        return self.node_map.keys()

    def __iter__(self):
        return self.node_map.keys().__iter__()

    def to_scipy_sparse_matrix(self):
        return coo_matrix(([1.]*len(self.edge_row), (self.edge_row, self.edge_col)),
                          shape=(len(self.node_map), len(self.node_map)), dtype=float).asformat("csr")

    def __len__(self):
        return len(self.node_map)

    def number_of_edges(self):
        return len(self.edge_row)

    def __contains__(self, node):
        return node in self.node_map
