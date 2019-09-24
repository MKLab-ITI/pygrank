class Tautology:
    def __init__(self):
        pass

    def rank(self, ranks):
        return ranks


class Normalize:
    def __init__(self, ranker=None):
        self.ranker = Tautology() if ranker is None else ranker

    def rank(self, personalization):
        ranks = self.ranker.rank(personalization)
        max_rank = max(ranks.values())
        return {node: rank / max_rank for node, rank in ranks.items()}


class NonParametric:
    def __init__(self, ranker=None):
        self.ranker = Tautology() if ranker is None else ranker

    def rank(self, personalization):
        ranks = self.ranker.rank(personalization)
        max_rank = max(ranks.values())
        return {node: rank / max_rank for node, rank in ranks.items()}