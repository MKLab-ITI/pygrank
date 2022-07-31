from scipy.sparse import coo_matrix


class Graph:
    def __init__(self, directed=False):
        self.edge_row = list()
        self.edge_col = list()
        self.node_map = dict()
        self.directed = directed
        self._adj = None
        self._degrees = None
        self._masked_out = None
        self._non_maksed_edge_num = None

    def add_node(self, node):
        if node not in self.node_map:
            self.node_map[node] = len(self.node_map)
        return self.node_map[node]

    def remove_edge(self, u, v):
        self._non_maksed_edge_num = None
        if self._adj is not None:
            if u in self._adj:
                self._adj[u].remove(v)
            if not self.directed and v in self._adj:
                self._adj[v].remove(u)
        u = self.node_map[u]
        v = self.node_map[v]
        if self._masked_out is None:
            self._masked_out = dict()
        if u not in self._masked_out:
            self._masked_out[u] = set()
        self._masked_out[u].add(v)
        if not self.directed:
            if v not in self._masked_out:
                self._masked_out[v] = set()
            self._masked_out[v].add(u)

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
        self._non_maksed_edge_num = None
        if self._masked_out is not None:
            if u in self._masked_out and v in self._masked_out[u]:
                self._masked_out[u].remove(v)
            if not self.directed and v in self._masked_out and u in self._masked_out[u]:
                self._masked_out[v].remove(u)
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

    def to_scipy_sparse_array(self):
        if self._masked_out:
            return coo_matrix(([0 if u in self._masked_out and v in self._masked_out[u] else 1 for u, v in zip(self.edge_row, self.edge_col)], (self.edge_row, self.edge_col)),
                              shape=(len(self.node_map), len(self.node_map)), dtype=float)#.asformat("csr")
        return coo_matrix(([1.]*len(self.edge_row), (self.edge_row, self.edge_col)),
                          shape=(len(self.node_map), len(self.node_map)), dtype=float)#.asformat("csr")

    def __len__(self):
        return len(self.node_map)

    def number_of_nodes(self):
        return len(self.node_map)

    def number_of_edges(self):
        if self._masked_out is not None:
            if self._non_maksed_edge_num is None:
                self._non_maksed_edge_num = sum(0 if u in self._masked_out and v in self._masked_out[u] else 1 for u, v in zip(self.edge_row, self.edge_col))
            ret = self._non_maksed_edge_num
        else:
            ret = len(self.edge_row)
        return ret if self.directed else ret / 2

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
            if self._masked_out is not None:
                for u, v in zip(self.edge_row, self.edge_col):
                    if u in self._masked_out and v in self._masked_out[u]:
                        continue
                    u = inverse_map[u]
                    v = inverse_map[v]
                    if u not in self._adj:
                        self._adj[u] = set()
                    self._adj[u].add(v)
            else:
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
