from experiments.importer import import_SNAP
from pygrank.algorithms import PageRank, HeatKernel, AbsorbingWalks
from pygrank.algorithms.postprocess import Tautology, SeedOversampling, BoostedSeedOversampling
from pygrank.algorithms.utils import preprocessor, to_signal
from pygrank.measures.utils import split_groups
from pygrank.measures import AUC


def perc(num):
    if num<0.005:
        return "0"
    ret = str(int(num*100+.5)/100.) # helper method to pretty print percentages
    if len(ret) < 4:
        ret += "0"
    if ret[0] == "0":
        return ret[1:]
    return ret


def fill(algorithm="", chars=14):
    return algorithm+(" "*(chars-len(algorithm)))


def supervised_benchmark(algorithms, datasets, metric=AUC, delimiter=" \t ", endline=""):
    print("Comparing algorithms based on "+metric.__class__.__name__)
    print(delimiter.join([fill()]+[fill(algorithm) for algorithm in algorithms])+endline)
    datasets = [(dataset, 0) if len(dataset) != 2 else dataset for dataset in datasets]
    last_loaded_dataset = None
    for dataset, group_id in datasets:
        dataset_results = fill(dataset)
        if last_loaded_dataset != dataset:
            G, groups = import_SNAP(dataset, max_group_number=1+max(group_id for dat, group_id in datasets if dat == dataset))
            last_loaded_dataset = dataset
        group = set(groups[group_id])
        training, evaluation = split_groups(list(group), training_samples=0.1)
        training, evaluation = to_signal(G,{v: 1 for v in training}), to_signal(G,{v: 1 for v in evaluation})
        for algorithm in algorithms.values():
            dataset_results += delimiter+fill(perc(metric(evaluation, exclude=training)(algorithm.rank(G, training))))
            #print(algorithm.ranker.convergence)
        print(dataset_results+endline)


def create_variations(algorithms, variations):
    all = dict()
    for variation in variations:
        for algorithm in algorithms:
            all[algorithm+variation] = variations[variation](algorithms[algorithm])
    return all


datasets = ["ant","citeseer","pubmed","squirel", ("amazon", 0), ("amazon", 1), "dblp"]
pre = preprocessor(assume_immutability=True, normalization="symmetric")
algorithms = {
    "ppr0.85": PageRank(alpha=0.85, to_scipy=pre, max_iters=1000000, tol=1.E-9),
    "ppr0.99": PageRank(alpha=0.99, to_scipy=pre, max_iters=1000000, tol=1.E-9),
    "absorb": AbsorbingWalks(to_scipy=pre, max_iters=1000000, tol=1.E-6),
    #"Lhk3": HeatKernel(t=3, to_scipy=pre, krylov_dims=5, max_iters=1000000, tol=1.E-9),
    #"Lhk7": HeatKernel(t=7, to_scipy=pre, krylov_dims=5, max_iters=1000000, tol=1.E-9),
    "hk3": HeatKernel(t=3, to_scipy=pre, max_iters=1000000, tol=1.E-9),
    "hk7": HeatKernel(t=5, to_scipy=pre, max_iters=1000000, tol=1.E-9),
}
algorithms = create_variations(algorithms, {"": Tautology, "+SO": SeedOversampling})

supervised_benchmark(algorithms, datasets)


