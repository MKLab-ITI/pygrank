import pygrank as pg

hk5 = pg.HeatKernel(t=5, normalization="symmetric", renormalize=True)
hk5_advanced = hk5 >> pg.SeedOversampling() >> pg.Sweep() >> pg.Normalize()

_, graph, community = next(pg.load_datasets_one_community(["eucore"]))
personalization = {node: 1.0 for node in community}
scores = hk5_advanced(graph, personalization)
print(scores)

measure = pg.Conductance()
pg.benchmark_print_line("My algorithm's conductance", measure(scores))
print("Cite this algorithm as:", hk5_advanced.cite())
