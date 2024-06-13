# Link Prediction

## 1. Load nx graph
We are going to create a `networkx` graph. This cannot be of other 
types, such as
`pygrank.fastgraph` because, they do not implement the `remove_edge` method.
Doing this with automatic loaders consists of providing the library
from which the graph generation object will be obtained.

```python
import pygrank as pg
import networkx as nx

graph = next(pg.load_datasets_graph(["citeseer"], graph_api=nx))
```

## 2. Define predictor

We now define a link predictor on a graph for a node.
For the actual predictions, we will get the top scored nodes
to link to. We put last the already linked nodes, and eventually
keep only the top ones.

```python
def link_prediction(graph, node, top=5):
    algorithm = pg.PageRank(0.95) >> pg.SeedOversampling()  # or try pg.HeatKernel(5) >> pg.SeedOversampling("neighbors")
    ranks = algorithm(graph, {node: 1})
    return sorted(graph, key=lambda v: -ranks[v] if not graph.has_edge(node, v) else 0)[:top]
```

## 3. Evaluate

To evaluate the link prediction strategy we obtain a test subset of
each node's neighbors and then temporarily remove some of them with
`remove_edge`. Then, the recommendation takes place and precisions
and recalls are hand-computed. Finally, the removed edges are
re-insered before visiting the next node. 

```python
precisions = list()
recalls = list()
for node in graph:
    if graph.degree(node) < 10:
        continue
    _, test = pg.split(list(graph.neighbors(node)))
    test = set(test)
    for v in test:
        graph.remove_edge(node, v)
    recommendation = link_prediction(graph, node)
    TP = len([v for v in recommendation if v in test])
    precisions.append(TP/len(recommendation))
    recalls.append(TP/len(test))
    for v in test:
        graph.add_edge(node, v)
        
print("Avg. precision", sum(precisions)/len(precisions))  # 0.12631578947368416
print("Avg. recall", sum(recalls)/len(recalls))  # 0.22215956558061817
```