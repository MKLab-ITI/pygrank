class Tautology:
    def __init__(self):
        pass

    def rank(self, _, personalization):
        return personalization


class Normalize:
    def __init__(self, ranker=None):
        self.ranker = Tautology() if ranker is None else ranker

    def rank(self, G, personalization):
        ranks = self.ranker.rank(G, personalization)
        max_rank = max(ranks.values())
        return {node: rank / max_rank for node, rank in ranks.items()}


class Ordinals:
    def __init__(self, ranker=None):
        self.ranker = Tautology() if ranker is None else ranker

    def rank(self, G, personalization):
        ranks = self.ranker.rank(G, personalization)
        return {v: ord+1 for ord, v in enumerate(sorted(ranks, key=ranks.get, reverse=False))}