from scipy.sparse import coo_matrix


class Graph:
    def __init__(self, directed=False):
        self.edge_row = list()
        self.edge_col = list()
        self.node_map = dict()
        self.directed = directed
        self._adj = None
        self._degrees = None

    def add_node(self, node):
        if node not in self.node_map:
            self.node_map[node] = len(self.node_map)
        return self.node_map[node]

    def add_edge(self, u, v):
        if self._adj is not None:
            if u not in self._adj:
                self._adj[u] = set()
            self._adj[u].add(v)
            self._degrees[u] = len(self._adj[u])
            if not self.directed:
                if v not in self._adj:
                    self._adj[v] = set()
                self._adj[v].add(u)
                self._degrees[v] = len(self._adj[v])
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

    def number_of_nodes(self):
        return len(self.node_map)

    def number_of_edges(self):
        return len(self.edge_row)

    def copy(self):
        graph = Graph(self.directed)
        graph.node_map = {u: v for u, v in self.node_map.items()}
        graph.edge_row = [u for u in self.edge_row]
        graph.edge_col = [u for u in self.edge_col]
        return graph

    def __contains__(self, node):
        return node in self.node_map

    def _create_adjacency(self):
        if self._adj is None:
            inverse_map = {u: v for v, u in self.node_map.items()}
            self._adj = dict()
            for u, v in zip(self.edge_row, self.edge_col):
                u = inverse_map[u]
                v = inverse_map[v]
                if u not in self._adj:
                    self._adj[u] = set()
                self._adj[u].add(v)
            self._degrees = {u: len(self._adj[u]) if u in self._adj else 0 for u in self.node_map}

    def has_edge(self, u, v):
        self._create_adjacency()
        return u in self._adj and v in self._adj[u]

    @property
    def degree(self):
        self._create_adjacency()
        return self._degrees

    def neighbors(self, u):
        self._create_adjacency()
        return self._adj[u] if u in self._adj else set()
