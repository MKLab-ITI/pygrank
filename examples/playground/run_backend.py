import pygrank as pg
import torch
from timeit import default_timer as time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, graph, community = next(
    pg.load_datasets_one_community(["amazon"], graph_api=pg, min_group_size=50)
)
print(f"Nodes {len(graph)}, edges {graph.number_of_edges()}")

ppr = pg.HeatKernel(
    t=10,
    normalization="symmetric",
    assume_immutability=True,
    max_iters=20,
    error_type="iters",
)  # >> pg.Sweep()
signal = pg.to_signal(graph, {node: 1.0 for node in community})
preprocessor = ppr.preprocessor

for _ in range(2):
    print("-----------------------------------------------")
    with pg.Backend("numpy"):
        tic = time()
        for _ in range(1):
            scores = ppr(signal)
        # print(pg.sum(scores))
        print("numpy", ppr.convergence, "actual time", time() - tic)

    with pg.Backend("pytorch", device=device, mode="sparse"):
        tic = time()
        for _ in range(1):
            scores = ppr(signal)
        # print(pg.sum(scores))
        print("pytorch", ppr.convergence, "actual time", time() - tic)

    with pg.Backend("torch_sparse", device=device):
        tic = time()
        for _ in range(1):
            scores = ppr(signal)
        # print(pg.sum(scores))
        print("torch_sparse", ppr.convergence, "actual time", time() - tic)
