import pygrank as pg

_, graph, group = pg.load_one("eucore")
signal = pg.to_signal(graph, group)

train, test = pg.split(signal, training_samples=0.5)
train, valid = pg.split(train, training_samples=0.5)

tuner = pg.ParameterTuner(
    lambda params: pg.PageRank(alpha=params[0]),
    max_vals=[0.99],
    min_vals=[0.5],
    fraction_of_training=1,
    measure=pg.NDCG(valid, exclude=train + test).as_supervised_method(),
)

scores_pagerank = pg.PageRank(max_iters=1000)(train)
scores_tuned = tuner(train)

measure = pg.AUC(test, exclude=train + valid)
pg.benchmark_print_line("Pagerank", measure(scores_pagerank))
pg.benchmark_print_line("Tune", measure(scores_tuned))
