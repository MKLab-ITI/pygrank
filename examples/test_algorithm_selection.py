import pygrank as pg
_, graph, communities = next(pg.load_datasets_multiple_communities(["EUCore"], max_group_number=3))
train, test = pg.split(communities, 0.05)  # 5% of community members are known
algorithms = pg.create_variations(pg.create_demo_filters(), pg.Normalize)

supervised_algorithm = pg.AlgorithmSelection(algorithms.values(), measure=pg.AUC)
print(supervised_algorithm.cite())
modularity_algorithm = pg.AlgorithmSelection(algorithms.values(), fraction_of_training=1, measure=pg.Modularity().as_supervised_method())

linkauc_algorithm = None
best_evaluation = 0
linkAUC = pg.LinkAssessment(graph, similarity="cos", hops=1)  # LinkAUC, because emails systemically exhibit homophily
for algorithm in algorithms.values():
    evaluation = linkAUC.evaluate({community: algorithm(graph, seeds) for community, seeds in train.items()})
    if evaluation > best_evaluation:
        best_evaluation = evaluation
        linkauc_algorithm = algorithm

supervised_aucs = list()
modularity_aucs = list()
linkauc_aucs = list()
for seeds, members in zip(train.values(), test.values()):
    measure = pg.AUC(members, exclude=seeds)
    supervised_aucs.append(measure(supervised_algorithm(graph, seeds)))
    modularity_aucs.append(measure(modularity_algorithm(graph, seeds)))
    linkauc_aucs.append(measure(linkauc_algorithm(graph, seeds)))

print("Supervised", sum(supervised_aucs) / len(supervised_aucs))
print("Modularity", sum(modularity_aucs)/len(modularity_aucs))
print("LinkAUC", sum(modularity_aucs)/len(modularity_aucs))





