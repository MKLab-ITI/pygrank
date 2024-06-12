import pygrank as pg

_, graph, communities = next(pg.load_datasets_multiple_communities(["dblp"]))
algorithm = pg.PageRank(
    alpha=0.9, assume_immutability=True
)  # cache graph preprocessing

comm_scores = {name: algorithm(graph, members) for name, members in communities.items()}

import tqdm  # install this to be able to set it as a progress bar argument below

measure = pg.LinkAssessment(graph, progress=tqdm.tqdm)
print(measure(comm_scores))
