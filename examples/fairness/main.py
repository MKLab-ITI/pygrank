import pygrank as pg
import setting

pg.split()

print("----------- PPR.85 -----------")
filter = pg.PageRank(0.85, max_iters=10000, tol=1.E-9, assume_immutability=True, normalization="symmetric")
setting.experiment(filter, ["citeseer"], pg.AUC, [.1, .2, .3, .4, .5], fix_personalization=True, repeats=2, sensitive_group=100)
