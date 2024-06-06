import pygrank as pg
from timeit import default_timer as time


if __name__ == '__main__':
    with pg.Backend("dask", splits=4, n_workers=4):
        _, graph, community = next(pg.load_datasets_one_community(["amazon"]))
        ppr = pg.PageRank(alpha=0.9, normalization="symmetric", assume_immutability=True)
        ppr.preprocessor(graph)
        ppr_advanced = ppr >> pg.SeedOversampling() >> pg.Sweep() >> pg.Normalize()

        tic = time()
        personalization = {node: 1.0 for node in community}
        scores = ppr_advanced(graph, personalization)
        #print(scores)

        measure = pg.Conductance()
        pg.benchmark_print_line("My algorithm's conductance", measure(scores))
        print("Cite this algorithm as:", ppr_advanced.cite())
        print("ETA", time()-tic)