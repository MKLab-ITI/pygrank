import pygrank as pg
from tqdm import tqdm


class StochasticSeedOversampling(pg.Postprocessor):
    def __init__(self, ranker=None):
        super().__init__(ranker)

    def rank(self,
             graph: pg.GraphSignalGraph = None,
             personalization: pg.GraphSignalData = None,
             **kwargs):
        personalization = pg.to_signal(graph, personalization)
        graph = personalization.graph
        ranks = self.ranker(personalization)
        ret = 0
        total_sum = pg.sum(ranks)
        accum_sum = 0
        for threshold in sorted(ranks.values()):
            accum_sum += threshold
            if accum_sum > total_sum*0.1:
                break
        for i, v in enumerate(ranks):
            pg.utils.log(f"{i}/{len(ranks)}")
            if ranks[v] >= threshold:
                partial = ranks >> pg.Threshold(ranks[v], inclusive=True) >> self.ranker
                ret = partial * ranks[v] + ret
        return ret


algs = {"ppr": pg.PageRank(0.9),
        "ppr+so": pg.PageRank(0.9) >> pg.SeedOversampling(),
        "ppr+bso": pg.PageRank(0.9) >> pg.BoostedSeedOversampling(),
        "ppr+sso": pg.PageRank(0.9) >> StochasticSeedOversampling(),
        }

loader = pg.load_datasets_one_community(["citeseer"])
pg.benchmark_print(pg.benchmark(algs, loader, pg.AUC, 3))
