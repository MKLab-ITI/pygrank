import pygrank as pg


_, graph, group = next(pg.load_datasets_one_community(["graph9"]))
algorithm = pg.PageRank(tol=0.001)
with pg.Backend("matvec"):
    ranks = algorithm(graph, {v: 1 for v in group})
with pg.Backend("numpy"):
    ranks2 = algorithm(graph, {v: 1 for v in group})

print(ranks["A"], ranks2["A"])