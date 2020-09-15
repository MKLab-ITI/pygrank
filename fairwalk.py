from pygrank.algorithms.pagerank import AbsorbingRank, PageRank, HeatKernel
from pygrank.algorithms.postprocess import Normalize, FairPostprocessor, Sweep, FairPersonalizer
from pygrank.metrics.utils import split_groups
from sklearn import metrics
import random


def perc(num):
    return int(num*100+.5)/100.


def experiments(algorithm, seed_size, dataset):
    random.seed(1)
    if "twitter" in dataset:
        import data.twitter_fairness.importer
        G, sensitive, labels = data.twitter_fairness.importer.load()
        repeats = 5
    elif "facebook" in dataset:
        import data.facebook_fairness.importer
        if "686" in dataset:
            G, sensitive, labels = data.facebook_fairness.importer.load("data/facebook_fairness/686")
        else:
            G, sensitive, labels = data.facebook_fairness.importer.load("data/facebook_fairness/0")
        repeats = 5
    else:
        raise Exception("Invalid dataset name")

    eps = 1.E-12
    p1 = sum([labels[v] for v in G if sensitive[v] == 0]) / (eps + sum([1 for v in G if sensitive[v] == 0]))
    p2 = sum([labels[v] for v in G if sensitive[v] == 1]) / (eps + sum([1 for v in G if sensitive[v] == 1]))
    #print("dataset pRule", min(p1,p2)/max(p1,p2), sum(sensitive.values())/len(G), sum(labels.values())/len(G), len(G), G.number_of_edges())

    accuracy = list()
    treatment = list()
    mistreatment = list()
    treatment_overtrained = list()
    for _ in range(repeats):
        training, evaluation = split_groups(list(G), fraction_of_training=seed_size)
        if "extreme" in dataset:
            ranks = algorithm.rank(G, {v: labels[v] for v in training if sensitive[v]==0}, sensitive=sensitive)
        else:
            ranks = algorithm.rank(G, {v: labels[v] for v in training}, sensitive=sensitive)
        p1 = sum([ranks[v] for v in evaluation if sensitive[v] == 0]) / (eps+sum([1 for v in evaluation if sensitive[v] == 0]))
        p2 = sum([ranks[v] for v in evaluation if sensitive[v] == 1]) / (eps+sum([1 for v in evaluation if sensitive[v] == 1]))
        fpr, tpr, thresholds = metrics.roc_curve([labels[v] for v in evaluation], [ranks[v] for v in evaluation])
        accuracy.append(metrics.auc(fpr, tpr))
        treatment.append(min(p1,p2)/(eps+max(p1,p2)))

        p1 = sum([ranks[v] for v in G if sensitive[v] == 0]) / (eps+sum([1 for v in G if sensitive[v] == 0]))
        p2 = sum([ranks[v] for v in G if sensitive[v] == 1]) / (eps+sum([1 for v in G if sensitive[v] == 1]))
        treatment_overtrained.append(min(p1,p2)/(eps+max(p1,p2)))

        fpr1 = sum([ranks[v] for v in evaluation if sensitive[v] == 0 and labels[v]==0])/(eps+sum([1 for v in evaluation if sensitive[v] == 0 and labels[v]==0]))
        fpr2 = sum([ranks[v] for v in evaluation if sensitive[v] == 1 and labels[v]==0])/(eps+sum([1 for v in evaluation if sensitive[v] == 1 and labels[v]==0]))

        mistreatment.append(abs(fpr1-fpr2))
        #print('seeds', seed_size, 'AUC', int(sum(accuracy)/len(accuracy)*1000)/1000., '\tpRule', int(sum(treatment)/len(treatment)*1000)/1000., int(sum(treatment_overtrained)/len(treatment_overtrained)*1000)/1000., '\tDisp. Misreatment', int(sum(mistreatment)/len(mistreatment)*1000)/1000.)
    return sum(accuracy)/len(accuracy), sum(treatment)/len(treatment), sum(mistreatment)/len(mistreatment), sum(treatment_overtrained)/len(treatment_overtrained)


datasets = ["facebook 0", "facebook 686", "facebook 0 extreme", "facebook 686 extreme", "twitter extreme"]
for dataset in datasets:
    print('%', dataset)
    points = 10

    #ppr = PageRank(alpha=0.85, max_iters=10000, tol=1.E-9, assume_immutability=True, normalization="symmetric")
    ppr = HeatKernel(t=3, max_iters=10000, tol=1.E-9, assume_immutability=True, normalization="symmetric")
    seeds = [(1.+i)/points for i in range(points) if (1.+i)/points<=0.9 and (1.+i)/points>=0.1]
    seeds = seeds[:3]

    algorithms = {
                    #"FairSweep": Normalize(Sweep(Fair(ppr, "B"))),#FairSweep(ppr),
                    "None": ppr,
                    "Mult": FairPostprocessor(ppr, "B"),
                    "LFRPO": FairPostprocessor(ppr, "O"),
                    "Sweep": Normalize(Sweep(ppr)),
                    "FP": Normalize(FairPersonalizer(ppr)),
                    "CFP": Normalize(FairPersonalizer(ppr, .80,pRule_weight=10)),
                    "SweepLFRPO": Normalize(FairPostprocessor(Sweep(ppr), "O")),
                    "SweepFP": Normalize(FairPersonalizer(Sweep(ppr))),
                    "SweepCFP": Normalize(FairPersonalizer(Sweep(ppr),.80,pRule_weight=10)),
                    #"FPSweep": Normalize(Sweep(PersonalizationFair(ppr))),
                    #"CFPSweep": Normalize(Sweep(PersonalizationFair(ppr,.80,retain_rank_weight=.1))),
                  }

    #print("clear all\nfigure(1);\nclf\nseeds = ["+",".join([str(seed) for seed in seeds])+"];\n")
    for name, algorithm in algorithms.items():
        aucs = list()
        prules = list()
        prules_global = list()
        for seed in seeds:
            auc, prule, mistreatment, prule_global = experiments(algorithm, seed, dataset)
            aucs.append(perc(auc))
            prules.append(perc(prule))
            prules_global.append(perc(prule_global))
        #print(name+"_auc = ["+",".join([str(auc) for auc in aucs)+"]; %", perc(sum(aucs)/len(aucs)), perc(sum(aucs[:3])/3))
        #print(name+"_prule = ["+",".join([str(prule) for prule in prules])+"];%", perc(sum(prules)/len(prules)), perc(sum(prules[:3])/3))
        print(name, "&", perc(sum(aucs)/len(aucs)), "&", min(perc(sum(prules)/len(prules)), perc(sum(prules_global)/len(prules_global))))

"""
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
"""