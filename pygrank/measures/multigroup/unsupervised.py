class MultiUnsupervised:
    def __init__(self, metric_type, *args, **kwargs):
        self.metric = metric_type(*args, **kwargs)

    def evaluate(self, scores):
        evaluations = [
            self.metric.evaluate(group_scores) for group_scores in scores.values()
        ]
        return sum(evaluations) / len(evaluations)

    def __call__(self, scores):
        return self.evaluate(scores)
