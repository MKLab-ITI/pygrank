class Tautology:
    """ Returns ranks as-are.

    This class can be used as a baseline against to compare other rank augmentation algorithms.
    """

    def __init__(self):
        pass

    def rank(self, _, personalization):
        return personalization


class Normalize:
    """ Normalizes ranks by dividing with their maximal value.

    Attributes:
        ranker: Optional. The ranking algorithm.
    """

    def __init__(self, ranker=None):
        """ Initializes the class with a base ranker instance.

        Attributes:
            ranker: The base ranker instance. A Tautology() ranker is created if None was specified.
        """
        self.ranker = Tautology() if ranker is None else ranker

    def rank(self, G, personalization):
        ranks = self.ranker.rank(G, personalization)
        max_rank = max(ranks.values())
        return {node: rank / max_rank for node, rank in ranks.items()}


class Ordinals:
    """ Converts ranking outcome to ordinal numbers.

     The highest rank is set to 1, the second highest to 2, etc.
     """

    def __init__(self, ranker=None):
        """ Initializes the class with a base ranker instance.

        Attributes:
            ranker: Optional. The base ranker instance. A Tautology() ranker is created if None was specified.
        """
        self.ranker = Tautology() if ranker is None else ranker

    def rank(self, G, personalization):
        ranks = self.ranker.rank(G, personalization)
        return {v: ord+1 for ord, v in enumerate(sorted(ranks, key=ranks.get, reverse=False))}