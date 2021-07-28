import networkx as nx
import random

def test_graph(directed=False):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "D")
    G.add_edge("E", "F")
    G.add_edge("F", "G")
    G.add_edge("G", "H")
    G.add_edge("H", "I")
    G.add_edge("I", "J")
    G.add_edge("J", "K")
    G.add_edge("A", "D")
    G.add_edge("B", "D")
    G.add_edge("B", "E")
    G.add_edge("E", "G")
    G.add_edge("G", "J")
    G.add_edge("G", "I")
    G.add_edge("H", "J")
    G.add_edge("I", "K")
    G.add_edge("L", "K")
    G.add_edge("K", "M")
    return G


def test_block_model_graph(nodes=600, seed=1):
    random.seed(1)
    G = nx.DiGraph()
    groups = [list(range(300)), list(range(300, nodes))]
    for i in range(nodes):
        for j in range(nodes):
            prob = 0.1 if (i<300)==(j<300) else 0.05
            if random.uniform(0, 1) < prob:
                G.add_edge(i, j)
    return G, groups