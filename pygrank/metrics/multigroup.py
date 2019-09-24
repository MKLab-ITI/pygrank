class LinkAUC:
    def __init__(self, G):
        self.G = G

    def evaluate(self, ranks):
        raise Exception("LinkAUC not yet implemented")


class MultiUnsupervised:
    def __init__(self, metric_type, G):
        self.metric = metric_type(G)

    def evaluate(self, ranks):
        evaluations = [self.metric.evaluate(group_ranks) for group_ranks in ranks.values()]
        return sum(evaluations) / len(evaluations)


class MultiSupervised:
    def __init__(self, metric_type, ground_truth):
        self.metrics = {group_id: metric_type(group_truth) for group_id, group_truth in ground_truth.items()}

    def evaluate(self, ranks):
        evaluations = [self.metrics[group_id].evaluate(group_ranks) for group_id, group_ranks in ranks.items()]
        return sum(evaluations) / len(evaluations)

