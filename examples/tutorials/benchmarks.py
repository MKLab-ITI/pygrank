import pygrank as pg

dataset_names = pg.downloadable_small_datasets()
print(
    dataset_names
)  # ['bigraph', 'blockmodel', 'citeseer', 'eucore', 'graph5', 'graph9']

algorithms = pg.create_demo_filters()
print(
    algorithms.keys()
)  # dict_keys(['PPR.85', 'PPR.9', 'PPR.99', 'HK3', 'HK5', 'HK7'])

loader = pg.load_datasets_one_community(dataset_names)
pg.benchmark_print(pg.benchmark(algorithms, loader, pg.AUC))
