from pygrank.algorithms import PageRank
from pygrank.algorithms import Normalize
from pygrank.measures.utils import split
import data.facebook_fairness.importer

G, sensitive, labels = data.facebook_fairness.importer.load("data/facebook_fairness/0")

algorithm = Normalize(PageRank())
training, evaluation = split(list(G), training_samples=0.1)
ranks = algorithm.rank(G, {v: labels[v] for v in training if sensitive[v] == 0}, sensitive=sensitive)


from pyvis.network import Network as VisualNetwork
vis = VisualNetwork(height=800, width=1200)
for v in G:
    #vis.add_node(v, color=(("#ff0000" if sensitive[v]==1 else 'rgb(0,255,0)') if labels[v]==1 else ("#00ff00" if sensitive[v]==1 else "#005500")) if v in G else "#000000")
    vis.add_node(v, color=("rgb("+str(int(255*ranks[v]))+",0,0)" if sensitive[v] == 1 else "rgb(0,"+str(int(255*ranks[v]))+",0)") if v in G else "#000000")
for u, v in G.edges():
    vis.add_edge(u, v)
vis.barnes_hut(
gravity=-80000,
central_gravity=0.3,
spring_length=10,
spring_strength=0.001,
damping=0.09,
overlap=0,
)

vis.show("example.html")