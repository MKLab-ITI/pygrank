import pygrank as pg
import networkx as nx

graph = nx.Graph()
graph.add_edge("A", "B")
graph.add_edge("A", "C")
graph.add_edge("C", "D")
graph.add_edge("D", "E")
signal = pg.to_signal(graph, {"A": 3, "C": 2})

algorithm = pg.HeatKernel(t=5) >> pg.Normalize("max") >> pg.Threshold(0.5)
scores = algorithm(signal)
print(scores)  # {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0, 'E': 0.0}

supervised = pg.Accuracy({"D": 1}, exclude=["A", "C"])
print(supervised(scores))  # 0.6666666666666667

unsupervised = pg.Density()  # lower is better
print(unsupervised(scores))  # 0.5
