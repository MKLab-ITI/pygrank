import pygrank as pg


algorithm = pg.HeatKernel(t=5,  # the number of hops to place maximal importance on
                         normalization="symmetric", renormalize=True)
algorithms = {"hk5": algorithm, "hk5+oversampling": pg.SeedOversampling(algorithm)}
algorithms = algorithms | pg.create_variations(algorithms, {"+sweep": pg.Sweep})
algorithms = pg.create_variations(algorithms, pg.Normalize)

_, graph, community = next(pg.load_datasets_one_community(["EUCore"]))
personalization = {node: 1. for node in community}  # missing scores considered zero
measure = pg.Conductance()  # smaller means tightly-knit stochastic community
for algorithm_name, algorithm in algorithms.items():
    scores = algorithm(graph, personalization)  # returns a dict-like pg.GraphSignal
    pg.benchmark_print_line(algorithm_name, measure(scores), tabs=[20, 5])  # pretty
