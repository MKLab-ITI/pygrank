import pygrank as pg

_, graph, group = next(pg.load_datasets_one_community(["citeseer"]))
alpha = 0.85

ranker = pg.PageRank(alpha=alpha, tol=1.0e-20, max_iters=1000) >> pg.Ordinals()
ranks = ranker(graph, group)
print(ranker.convergence)  # 70 iterations (0.012790299995685928 sec)
# print(ranks)  # {'100157': 107.0, '364207': 416.0, '38848': 626.0, 'bradshaw97introduction': 567.0, ... }

early_stop_ranker = (
    pg.PageRank(alpha=alpha, error_type=pg.OrderAccuracy) >> pg.Ordinals()
)
fastranks = early_stop_ranker(graph, group)
print(early_stop_ranker.convergence)  # 76 iterations (0.15961270000843797 sec)
print(pg.OrderAccuracy(ranks)(fastranks))  # 0.9996993385447985

convergence = pg.RankOrderConvergenceManager(pagerank_alpha=alpha, confidence=0.98)
smart_early_stop_ranker = (
    pg.PageRank(alpha=alpha, convergence=convergence) >> pg.Ordinals()
)
fastranks = smart_early_stop_ranker(graph, group)
print(smart_early_stop_ranker.convergence)  # 54 iterations (0.012241799995535985 sec)
print(pg.OrderAccuracy(ranks)(fastranks))  # 0.9882742032471438

convergence = pg.RankOrderConvergenceManager(
    pagerank_alpha=alpha, criterion="fraction_of_walks"
)
smart_early_stop_ranker = (
    pg.PageRank(alpha=alpha, convergence=convergence) >> pg.Ordinals()
)
fastranks = smart_early_stop_ranker(graph, group)
print(smart_early_stop_ranker.convergence)  # 0.9882742032471438
print(pg.OrderAccuracy(ranks)(fastranks))  # 0.7269993986770896

from matplotlib import pyplot as plt

plt.scatter(pg.to_numpy(ranks.np), pg.to_numpy(fastranks.np), marker=".")
plt.show()
