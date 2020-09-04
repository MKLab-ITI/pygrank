from data.facebook_fairness.importer import load
from pygrank.algorithms.pagerank import AbsorbingRank, PageRank
from pygrank.algorithms.postprocess import Normalize, Threshold, Fair
from pygrank.algorithms.filter import LanczosFilter
from pygrank.metrics.utils import split_groups
from sklearn import metrics

G, sensitive, labels = load()#"data/facebook_fairness/0")
G = G.to_directed()

p1 = sum([labels[v] for v in labels if sensitive[v] == 0]) / sum([1 for v in labels if sensitive[v] == 0])
p2 = sum([labels[v] for v in labels if sensitive[v] == 1]) / sum([1 for v in labels if sensitive[v] == 1])

datasetPrule = min(p1, p2) / max(p1, p2)

print('Dataset p-Rule', datasetPrule)


accuracy = list()
treatment = list()
mistreatment = list()

eps = 1.E-12
#algorithm = Threshold("none", Normalize(Fair(PageRank(alpha=0.85, max_iters=10000, tol=1.E-6, assume_immutability=True, normalization="col"), "none")))
algorithm = LanczosFilter(normalization="symmetric")

for _ in range(100):
    attraction = {v: (1 if sensitive[v]==0 else 1) for v in G}
    #absorption = {v: (1 if sensitive[v]==0 else 1) for v in G}
    #absorption = {v: (G.degree[v] if sensitive[v]==0 else G.degree[v]) for v in G}
    positive_label = 1
    seeds, _ = split_groups([v for v in G if labels[v]==positive_label], fraction_of_training=0.5)
    evaluation = [v for v in G if v not in seeds]
    ranks = algorithm.rank(G, {v: 1 for v in G}, sensitive=sensitive)
    negative_label = 1-positive_label
    p1 = sum([ranks[v] for v in evaluation if sensitive[v] == 0]) / sum([1 for v in evaluation if sensitive[v] == 0])
    p2 = sum([ranks[v] for v in evaluation if sensitive[v] == 1]) / sum([1 for v in evaluation if sensitive[v] == 1])
    if max(p1,p2)==0:
        continue
    fpr, tpr, thresholds = metrics.roc_curve([1. if labels[v]==positive_label else 0 for v in evaluation], [ranks[v] for v in evaluation])
    accuracy.append(metrics.auc(fpr, tpr))
    #accuracy.append(sum([1-abs(ranks[v]-labels[v]) for v in evaluation])/len(evaluation))
    treatment.append(min(p1,p2)/max(p1,p2))
    print('discovered', sum(ranks.values()))


    fpr1 = sum([ranks[v] for v in evaluation if sensitive[v] == 0 and labels[v]==negative_label])/(eps+sum([1 for v in evaluation if sensitive[v] == 0 and labels[v]==negative_label]))
    fpr2 = sum([ranks[v] for v in evaluation if sensitive[v] == 1 and labels[v]==negative_label])/(eps+sum([1 for v in evaluation if sensitive[v] == 1 and labels[v]==negative_label]))

    mistreatment.append(abs(fpr1-fpr2))

    print('AUC', int(sum(accuracy)/len(accuracy)*1000)/1000., '\tDisp. Impact', int(sum(treatment)/len(treatment)*1000)/1000., '\tDisp. Misreatment', int(sum(mistreatment)/len(mistreatment)*1000)/1000.)