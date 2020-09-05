from data.twitter_fairness.importer import load
from pygrank.algorithms.pagerank import AbsorbingRank, PageRank
from pygrank.algorithms.postprocess import Normalize, Threshold, Fair
from pygrank.algorithms.filters import LanczosFilter, GraphFilter
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
#algorithm = Normalize(Fair(PageRank(alpha=0.85, max_iters=10000, tol=1.E-6, assume_immutability=True, normalization="symmetric"), "none"))
algorithm = Normalize(GraphFilter(weights=[0.15*0.85**n for n in range(30)], normalization="symmetric"))
#algorithm = GraphFilter(weights=None, normalization="symmetric")

for _ in range(100):
    attraction = {v: (1 if sensitive[v]==0 else 1) for v in G}
    #absorption = {v: (1 if sensitive[v]==0 else 1) for v in G}
    #absorption = {v: (G.degree[v] if sensitive[v]==0 else G.degree[v]) for v in G}
    training, evaluation = split_groups(list(G), fraction_of_training=0.5)
    ranks = algorithm.rank(G, {v: labels[v] for v in training}, sensitive=sensitive)
    p1 = sum([ranks[v] for v in evaluation if sensitive[v] == 0]) / sum([1 for v in evaluation if sensitive[v] == 0])
    p2 = sum([ranks[v] for v in evaluation if sensitive[v] == 1]) / sum([1 for v in evaluation if sensitive[v] == 1])
    if max(p1,p2)==0:
        continue
    fpr, tpr, thresholds = metrics.roc_curve([labels[v] for v in evaluation], [ranks[v] for v in evaluation])
    accuracy.append(metrics.auc(fpr, tpr))
    #accuracy.append(sum([1-abs(ranks[v]-labels[v]) for v in evaluation])/len([v for v in evaluation]))
    treatment.append(min(p1,p2)/max(p1,p2))


    fpr1 = sum([ranks[v] for v in evaluation if sensitive[v] == 0 and labels[v]==0])/(eps+sum([1 for v in evaluation if sensitive[v] == 0 and labels[v]==0]))
    fpr2 = sum([ranks[v] for v in evaluation if sensitive[v] == 1 and labels[v]==0])/(eps+sum([1 for v in evaluation if sensitive[v] == 1 and labels[v]==0]))

    mistreatment.append(abs(fpr1-fpr2))

    print('AUC', int(sum(accuracy)/len(accuracy)*1000)/1000., '\tDisp. Impact', int(sum(treatment)/len(treatment)*1000)/1000., '\tDisp. Misreatment', int(sum(mistreatment)/len(mistreatment)*1000)/1000.)