from pygrank.measures.measure import Measure


class Time(Measure):
    """An abstract class that can be passed to benchmark experiments to indicate that they should report running time
    of algorithms. Instances of this class have no functionality."""

    pass
