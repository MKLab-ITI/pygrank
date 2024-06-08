import pygrank as pg

ppro = pg.PageRank(alpha=0.9, max_iters=1000, tol=1.0e-9) >> pg.AdHocFairness("O")
lfpr = pg.LFPR(alpha=0.9, max_iters=1000, tol=1.0e-9)

_, graph, groups = next(
    pg.load_datasets_multiple_communities(
        ["citeseer"], max_group_number=2, directed=False
    )
)
measure = pg.pRule(sensitive=groups[1])

pg.benchmark_print_line("ppro", ppro(graph, groups[0], sensitive=groups[1]) >> measure)
pg.benchmark_print_line("lfpr", lfpr(graph, groups[0], sensitive=groups[1]) >> measure)
