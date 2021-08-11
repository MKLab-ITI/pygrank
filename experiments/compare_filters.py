from pygrank.algorithms import PageRank, HeatKernel, GenericGraphFilter
from pygrank.algorithms import ParameterTuner
from pygrank.algorithms import Tautology, SeedOversampling
from pygrank.algorithms import preprocessor
from pygrank.measures import AUC
from pygrank import benchmark

datasets = ["ant", "citeseer", "pubmed", "squirel"]
pre = preprocessor(assume_immutability=True, normalization="symmetric")
algorithms = {
    "ppr0.85": PageRank(alpha=0.85, to_scipy=pre, max_iters=1000000, tol=1.E-9),
    "ppr0.99": PageRank(alpha=0.99, to_scipy=pre, max_iters=1000000, tol=1.E-9),
    "hk3": HeatKernel(t=3, to_scipy=pre, max_iters=1000000, tol=1.E-9),
    "hk7": HeatKernel(t=5, to_scipy=pre, max_iters=1000000, tol=1.E-9),
    "tuned": ParameterTuner(to_scipy=pre, max_iters=1000000, tol=1.E-9),
}
#algorithms = benchmark.create_variations(algorithms, {"": Tautology, "+SO": SeedOversampling})
loader = benchmark.dataset_loader(datasets)
benchmark.supervised_benchmark(algorithms, loader, AUC, verbose=True)
