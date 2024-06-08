from pygrank.core import GraphSignalData


class Measure(object):
    def __call__(self, scores: GraphSignalData):
        return self.evaluate(scores)

    def evaluate(self, scores: GraphSignalData):
        raise Exception(
            "Non-abstract subclasses of Measure should implement an evaluate method"
        )

    def best_direction(self) -> int:
        """
        Automatically determines if higher or lower values of the measure are better.
        Design measures so that outcomes of this method depends **only** on their class,
        as it follows a class-based hashing to guarantee speed. Otherwise override derived classes.

        Returns:
            1 if higher values of the measure are better, -1 otherwise.
        """
        raise Exception("Measure should implement a best_direction method")

    def as_supervised_method(self):
        def dummy_constructor(known_scores, exclude=None):
            self.known_scores = known_scores
            self.exclude = exclude
            return self

        dummy_constructor.__name__ = self.__class__.__name__
        return dummy_constructor

    def as_unsupervised_method(self):
        def dummy_constructor(graph=None):
            self.graph = graph
            return self

        dummy_constructor.__name__ = self.__class__.__name__
        return dummy_constructor

    def as_immutable_method(self):
        def dummy_constructor(*args, **kwargs):
            return self

        dummy_constructor.__name__ = self.__class__.__name__
        return dummy_constructor
