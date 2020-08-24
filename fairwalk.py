from data.facebook_fairness.importer import load
from pygrank.algorithms.pagerank import AbsorbingRank, PageRank
from pygrank.algorithms.postprocess import Normalize
import networkx as nx

G, sensitive, labels = load("data/facebook_fairness/0")
G = G.to_directed()


p1 = len([v for v in G if labels[v] == 0 and sensitive[v] == 0]) / len([v for v in G if sensitive[v] == 0])
p2 = len([v for v in G if labels[v] == 0 and sensitive[v] == 1]) / len([v for v in G if sensitive[v] == 1])
print(min(p1 / p2, p2 / p1))

treatment = list()
mistreatment = list()

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

for v in G:
    attraction = {v: (1 if sensitive[v]==0 else 1) for v in G}
    absorption = {v: (1 if sensitive[v]==0 else 1) for v in G}
    #absorption = {v: (G.degree[v] if sensitive[v]==0 else G.degree[v]) for v in G}
    ranks = Normalize(PageRank(alpha=0.85,max_iters=10000,tol=1.E-12)).rank(G, {v: 1})#, attraction=attraction, absorption=absorption)
    p1 = sum([ranks[v] for v in G if sensitive[v] == 0])
    p2 = sum([ranks[v] for v in G if sensitive[v] == 1])
    if p1==0 or p2==0:
        continue
    treatment.append(min(p1 / p2, p2 / p1))

    fpr1 = sum([ranks[v] for v in G if sensitive[v] == 0 and labels[v]==0])/sum([1 for v in G if sensitive[v] == 0 and labels[v]==0])
    fpr2 = sum([ranks[v] for v in G if sensitive[v] == 1 and labels[v]==0])/sum([1 for v in G if sensitive[v] == 1 and labels[v]==0])

    fnr1 = sum([ranks[v] for v in G if sensitive[v] == 0 and labels[v]==1])/sum([1 for v in G if sensitive[v] == 0 and labels[v]==1])
    fnr2 = sum([ranks[v] for v in G if sensitive[v] == 1 and labels[v]==1])/sum([1 for v in G if sensitive[v] == 1 and labels[v]==1])

    mistreatment.append(abs(fpr1-fpr2)+abs(fnr1-fnr2))

    print('Disp. Treatment', int(sum(treatment)/len(treatment)*1000)/1000., '\t Disp. Misreatment', int(sum(mistreatment)/len(mistreatment)*1000)/1000.)