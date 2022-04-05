import pygrank as pg
_, graph, community = next(pg.load_datasets_one_community(["EUCore"]))
algorithm = pg.HeatKernel(t=5, # the number of hops away HeatKernel places maximal importance on
                         normalization="symmetric", renormalize=True)
personalization = {node: 1. for node in community} # ignored nodes assumed to be zeroes
algorithms = {"HK5": algorithm, "HK5+Oversampling": pg.SeedOversampling(algorithm)}
algorithms = algorithms | pg.create_variations(algorithms, {"+Sweep": pg.Sweep})
algorithms = pg.create_variations(algorithms, {"": pg.Normalize})

measure = pg.Conductance()
for algorithm_name, algorithm in algorithms.items():
    scores = algorithm(graph, personalization) # returns a dict-like pg.GraphSignal
    print(algorithm_name, measure(scores))