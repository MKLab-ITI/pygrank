from experiments.importer import import_SNAP
from pygrank.algorithms import PageRank, HeatKernel
from pygrank.algorithms.utils import preprocessor, to_signal
from pygrank.metrics.utils import split_groups
from pygrank.metrics import Accuracy

def perc(num):
    if num<0.005:
        return "0"
    ret = str(int(num*100+.5)/100.) # helper method to pretty print percentages
    if len(ret) < 4:
        ret += "0"
    if ret[0] == "0":
        return ret[1:]
    return ret

def benchmark(algorithms, datasets, metric, delimiter=" \t ", endline=""):
    print(delimiter.join([" "*10] + list(algorithms.keys()))+endline)
    for dataset in datasets:
        dataset_results = dataset+" "*(10-len(dataset))
        G, groups = import_SNAP(dataset, max_group_number=1)
        group = set(groups[0])
        training, evaluation = split_groups(list(G), fraction_of_training=0.1)
        training, evaluation = to_signal(G,{v: 1 for v in training if v in group}), to_signal(G, {v: 1 for v in evaluation if v in group})
        for algorithm in algorithms.values():
            dataset_results += delimiter+perc(Accuracy(evaluation)(algorithm.rank(G, training)))+"     "
        print(dataset_results+endline)


pre = preprocessor(assume_immutability=True, normalization="auto")
algorithms = {
    "ppr0.85": PageRank(alpha=0.85, to_scipy=pre, max_iters=1000000, tol=1.E-6, assume_immutability=True),
    "ppr0.99": PageRank(alpha=0.99, to_scipy=pre, max_iters=1000000, tol=1.E-6, assume_immutability=True),
    "hk3    ": HeatKernel(t=3, to_scipy=pre, max_iters=1000000, tol=1.E-9, assume_immutability=True),
    "hk7    ": HeatKernel(t=7, to_scipy=pre, max_iters=1000000, tol=1.E-9, assume_immutability=True),
}
datasets = ["amazon", "citeseer","pubmed","squirel"]
benchmark(algorithms, datasets, Accuracy)
#for dataset in datasets:



