import pygrank as pg
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with pg.Backend("torch_sparse", device=device):
    _, graph, community = next(pg.load_datasets_one_community(["amazon"]))
    ppr = pg.PageRank(
        alpha=0.9,
        normalization="symmetric",
        assume_immutability=True,
        convergence=pg.ConvergenceManager(max_iters=38, error_type="iters"),
    )
    ppr.preprocessor(graph)
    signal = pg.to_signal(graph, {node: 1.0 for node in community})
    torch.cuda.synchronize()  # correct timing
    scores = ppr(signal)
    print(ppr.convergence)
    print(scores["B00005MHUG"])  # 0.00508212111890316
    print(scores["B00006RGI2"])  # 0.70645672082901
    print(scores["0006497993"])  # 0.19633759558200836
