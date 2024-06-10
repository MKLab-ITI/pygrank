import pygrank as pg
import torch
from timeit import default_timer as time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, graph, community = next(pg.load_datasets_one_community(["youtube"], graph_api=pg, min_group_size=50))
print(f"Nodes {len(graph)}, edges {graph.number_of_edges()}")

ppr = pg.HeatKernel(
    normalization="symmetric",
    assume_immutability=True
)
signal = pg.to_signal(graph, {node: 1.0 for node in community})
preprocessor = ppr.preprocessor
#ppr = pg.ParameterTuner(preprocessor=preprocessor)
"""
with pg.Backend("numpy"):
    preprocessor(graph)
    torch.cuda.synchronize()  # correct timing
    tic = time()
    scores = ppr(signal)
    print("numpy", ppr.convergence, "actual time", time()-tic)"""

with pg.Backend("torch_sparse", device=device):
    preprocessor(graph)
    torch.cuda.synchronize()  # correct timing
    tic = time()
    scores = ppr(signal)
    print("torch_sparse", ppr.convergence, "actual time", time()-tic)