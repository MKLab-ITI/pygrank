import numpy as np
from pygrank.algorithms.autotune import optimize
from random import random
import time


class Personalizer:
    def __init__(self, ranker, recursions=1, kernels=2, tol=1.E-2):
        self.ranker = ranker
        self.recursions = recursions
        self.tol = tol
        self.kernels = kernels

    def __culep(self, personalization, original_personalization, ranks, params, rate=1):
        p = personalization*params[-1]
        for i in range(self.kernels):
            p += (rate*params[2*i])*np.exp((ranks/ranks.max())*(params[2*i+1])) # explicit computation order to minimize time
        p_max = p.max()
        if p_max <= 0:
            return personalization
        return p / p_max

    def rank(self, G, personalization, loss, training_personalization=None, *args, **kwargs):
        personalization = to_numpy(G, personalization)
        training_personalization = None if training_personalization is None else self.ranker.to_numpy(G, training_personalization)
        prev_loss = 0
        ranks = self.ranker.rank(G, personalization, *args, as_dict=False, **kwargs)
        training_ranks = ranks if training_personalization is None else self.ranker.rank(G, training_personalization, *args, as_dict=False, **kwargs)
        params = [1, 1]*self.kernels + [1]
        original_personalization = personalization
        original_training_personalization = training_personalization
        for _ in range(self.recursions):
            def param_loss(par):
                if sum(abs(p) for p in par) == 0:
                    return float('inf')
                try:
                    pers = personalization if training_personalization is None else training_personalization
                    opers = original_personalization if training_personalization is None else original_training_personalization
                    return loss.evaluate(self.ranker.rank(G, self.__culep(pers, opers, training_ranks, par), *args, as_dict=False, **kwargs))
                except:
                    return float('inf')

            params = optimize(param_loss, [1, 10] * self.kernels + [1], min_vals=[-1, -10]*self.kernels+[0], partitions=5, deviation_tol=self.tol, divide_range=2, weights=params)
            personalization = self.__culep(personalization, original_personalization, ranks, params)
            training_personalization = None if training_personalization is None else self.__culep(training_personalization, original_personalization, training_ranks, params)
            ranks = self.ranker.rank(G, personalization, *args, as_dict=False, **kwargs)
            training_ranks = ranks if training_personalization is None else self.ranker.rank(G, training_personalization, *args, as_dict=False, **kwargs)
            if self.recursions != 1:
                curr_loss = loss.evaluate(ranks)
                print(curr_loss, params)
                if curr_loss >= prev_loss:
                    break
                prev_loss = curr_loss
            best_ranks = ranks
        return dict(zip(G.nodes(), map(float, best_ranks)))
