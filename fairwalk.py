from pygrank.algorithms.pagerank import AbsorbingRank, PageRank
from pygrank.algorithms.postprocess import Normalize, Fair, Sweep, CULEP
from pygrank.metrics.utils import split_groups
from sklearn import metrics
import random


def perc(num):
    return int(num*100)/100.


def experiments(algorithm, seed_size, dataset):
    random.seed(1)
    if dataset=="twitter":
        import data.twitter_fairness.importer
        G, sensitive, labels = data.twitter_fairness.importer.load()
        repeats = 5
    elif dataset=="facebook":
        import data.facebook_fairness.importer
        G, sensitive, labels = data.facebook_fairness.importer.load()
        repeats = 20
    else:
        raise Exception("Invalid dataset name")
    accuracy = list()
    treatment = list()
    mistreatment = list()
    eps = 1.E-12
    for _ in range(repeats):
        training, evaluation = split_groups(list(G), fraction_of_training=seed_size)
        #ranks = algorithm.rank(G, {v: labels[v] for v in training}, sensitive=sensitive)
        ranks = algorithm.rank(G, {v: labels[v] for v in training if sensitive[v]==0}, sensitive=sensitive)
        p1 = sum([ranks[v] for v in evaluation if sensitive[v] == 0]) / (eps+sum([1 for v in evaluation if sensitive[v] == 0]))
        p2 = sum([ranks[v] for v in evaluation if sensitive[v] == 1]) / (eps+sum([1 for v in evaluation if sensitive[v] == 1]))
        fpr, tpr, thresholds = metrics.roc_curve([labels[v] for v in evaluation], [ranks[v] for v in evaluation])
        accuracy.append(metrics.auc(fpr, tpr))
        treatment.append(min(p1,p2)/(eps+max(p1,p2)))

        fpr1 = sum([ranks[v] for v in evaluation if sensitive[v] == 0 and labels[v]==0])/(eps+sum([1 for v in evaluation if sensitive[v] == 0 and labels[v]==0]))
        fpr2 = sum([ranks[v] for v in evaluation if sensitive[v] == 1 and labels[v]==0])/(eps+sum([1 for v in evaluation if sensitive[v] == 1 and labels[v]==0]))

        mistreatment.append(abs(fpr1-fpr2))
        #print('seeds', seed_size, 'AUC', int(sum(accuracy)/len(accuracy)*1000)/1000., '\tpRule', int(sum(treatment)/len(treatment)*1000)/1000., '\tDisp. Misreatment', int(sum(mistreatment)/len(mistreatment)*1000)/1000.)
    return sum(accuracy)/len(accuracy), sum(treatment)/len(treatment), sum(mistreatment)/len(mistreatment)


dataset = "twitter"
points = 50

ppr = PageRank(alpha=0.85, max_iters=10000, tol=1.E-9, assume_immutability=True, normalization="col")
ppr_fair = PageRank(alpha=0.85, max_iters=10000, tol=1.E-9, assume_immutability=True, normalization="col", use_quotient=Normalize("sum", Fair("B")))

algorithms = {
                "CULEPSweep": Normalize(CULEP(Sweep(ppr))),
                "FairSweep": Normalize(Sweep(Fair(ppr, "B"))),#FairSweep(ppr),
                "None": ppr,
                "Redistribute": Fair(ppr, "O"),
                "Renormalize": Fair(ppr, "B"),
                "Sweep": Normalize(Sweep(ppr)),
                "RedistributeSweep": Normalize(Fair(Sweep(ppr), "O")),
              }

seeds = [(1.+i)/points for i in range(points) if (1.+i)/points<=0.5]
seeds.append(1)
print("clear all\nfigure(1);\nclf\nseeds = ["+",".join([str(seed) for seed in seeds[:-1]])+"];\n")
for name, algorithm in algorithms.items():
    aucs = list()
    prules = list()
    for seed in seeds:
        auc, prule, mistreatment = experiments(algorithm, seed, dataset)
        aucs.append(perc(auc))
        prules.append(perc(prule))
    print(name+"_auc = ["+",".join([str(auc) for auc in aucs[:-1]])+"]; %", perc(sum(aucs)/len(aucs)))
    print(name+"_prule = ["+",".join([str(prule) for prule in prules[:-1]])+"];%", perc(sum(prules)/len(prules)))
print()
for name in algorithms:
    print("plot(seeds, "+name+"_auc)")
    print("hold on")
print("legend('"+"','".join(list(algorithms.keys()))+"')")

print("figure(2);\nclf\n")
for name in algorithms:
    print("plot(seeds, "+name+"_prule)")
    print("hold on")
print("legend('"+"','".join(list(algorithms.keys()))+"')")