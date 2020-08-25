from data.facebook_fairness.importer import load
from pygrank.algorithms.pagerank import AbsorbingRank, PageRank
from pygrank.algorithms.postprocess import Normalize
from pygrank.algorithms.oversampling import BoostedSeedOversampling
from pygrank.metrics.utils import split_groups
from sklearn import metrics

G, sensitive, labels = load("data/facebook_fairness/0")
labels = {v: labels[v] for v in G}
G = G.to_directed()


p1 = len([v for v in G if labels[v] == 0 and sensitive[v] == 0]) / len([v for v in G if sensitive[v] == 0])
p2 = len([v for v in G if labels[v] == 0 and sensitive[v] == 1]) / len([v for v in G if sensitive[v] == 1])
print(min(p1 / p2, p2 / p1))

"""
for v in G:
    sum0 = 0
    sum1 = 1
    for u in G.successors(v):
        if sensitive[v] == 0:
            sum0 += 1
        else:
            sum1 += 1
    for u in list(G.successors(v)):
        if sensitive[v] == 0:
            G.add_edge(v, u, weight=(sum0+sum1)/sum0)
        else:
            G.add_edge(v, u, weight=(sum0+sum1)/sum1)
"""

accuracy = list()
treatment = list()
mistreatment = list()

for _ in range(10):
    attraction = {v: (1 if sensitive[v]==0 else 1) for v in G}
    absorption = {v: (1 if sensitive[v]==0 else 1) for v in G}
    #absorption = {v: (G.degree[v] if sensitive[v]==0 else G.degree[v]) for v in G}
    positive_label = 1
    seeds, evaluation = split_groups([v for v in G if labels[v]==positive_label], fraction_of_training=0.2)
    evaluation.extend([v for v in G if labels[v]!=positive_label])
    ranks = Normalize(BoostedSeedOversampling(PageRank(alpha=0.5,max_iters=10000,tol=1.E-12))).rank(G, {v: 1 for v in seeds})#, attraction=attraction, absorption=absorption)
    negative_label = 1-positive_label


    p1 = sum([ranks[v] for v in evaluation if sensitive[v] == 0])
    p2 = sum([ranks[v] for v in evaluation if sensitive[v] == 1])

    fpr, tpr, thresholds = metrics.roc_curve([1. if labels[v]==positive_label else 0 for v in evaluation], [ranks[v] for v in evaluation])
    accuracy.append(metrics.auc(fpr, tpr))
    #accuracy.append(sum([1-abs(ranks[v]-labels[v]) for v in labels])/len(labels))
    treatment.append(min(p1 / p2, p2 / p1))


    fpr1 = sum([ranks[v] for v in evaluation if sensitive[v] == 0 and labels[v]==positive_label])/sum([1 for v in evaluation if sensitive[v] == 0 and labels[v]==positive_label])
    fpr2 = sum([ranks[v] for v in evaluation if sensitive[v] == 1 and labels[v]==positive_label])/sum([1 for v in evaluation if sensitive[v] == 1 and labels[v]==positive_label])

    fnr1 = sum([ranks[v] for v in evaluation if sensitive[v] == 0 and labels[v]==negative_label])/sum([1 for v in evaluation if sensitive[v] == 0 and labels[v]==negative_label])
    fnr2 = sum([ranks[v] for v in evaluation if sensitive[v] == 1 and labels[v]==negative_label])/sum([1 for v in evaluation if sensitive[v] == 1 and labels[v]==negative_label])

    mistreatment.append(abs(fpr1-fpr2)+abs(fnr1-fnr2))

    print('Acc', int(sum(accuracy)/len(accuracy)*1000)/1000., '\tDisp. Impact', int(sum(treatment)/len(treatment)*1000)/1000., '\tDisp. Misreatment', int(sum(mistreatment)/len(mistreatment)*1000)/1000.)