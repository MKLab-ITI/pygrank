import pygrank as pg

for name, graph, community in pg.load_datasets_one_community(["EUCore", "Amazon"]):
    print(name, "nodes:   ", len(graph))
    print(name, "edges:   ", graph.number_of_edges())
    print(name, "examples:", len(community))

algorithm = pg.HeatKernel(t=5,  # the number of hops away HeatKernel places maximal importance on
                          assume_immutability=True,  # memoization on adjacency matrix normalization for fast reruns
                          normalization="symmetric", renormalize=True)

algorithms = {"HK5": algorithm, "HK5+Oversample": pg.SeedOversampling(algorithm)}
algorithms = algorithms | pg.create_variations(algorithms, {"+Sweep": pg.Sweep})
algorithms = pg.create_variations(algorithms, pg.Normalize)

print("\nOverlapping community detection conductance (lower = better)")
pg.benchmark_print(pg.benchmark(algorithms, pg.load_datasets_one_community(["EUCore", "Amazon"]), pg.Conductance))

