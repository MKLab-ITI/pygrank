class MultiSupervised:
    def __init__(self, metric_type, ground_truth, exclude=None):
        self.metrics = {
            group_id: (
                metric_type(group_truth)
                if exclude is None
                else metric_type(group_truth, exclude[group_id])
            )
            for group_id, group_truth in ground_truth.items()
        }

    def evaluate(self, scores):
        evaluations = [
            self.metrics[group_id].evaluate(group_scores)
            for group_id, group_scores in scores.items()
        ]
        return sum(evaluations) / len(evaluations)

    def __call__(self, scores):
        return self.evaluate(scores)
