import scipy
import numpy as np
from scipy.stats import norm
import time


class ConvergenceManager:
    """ Used to keep previous iteration and generally manage convergence of variables. Graph filters
    automatically create instances of this class by passing on appropriate parameters.

    Supported error types:
        "mabs": mean absolute value of rank differences. Throws exception on iteration max_iters.
        "msqrt": mean square root error of rank differences. Throws exception on iteration max_iters.
        "iters": Stops at max_iters (without throwing an exception). Ignores the tol argument.
        "const": Stops after first iteration.


    Examples:
        >>> convergence = ConvergenceManager()
        >>> convergence.start()
        >>> var = None
        >>> while not convergence.has_converged(var):
        >>>     ...
        >>>     var = ...
    """

    def __init__(self, tol=1.E-6, error_type="mabs", max_iters=100):
        """
        Initializes a convergence manager with a provided tolerance level, error type and number of iterations.

        Args:
            tol: Numerical tolerance to determine the stopping point (algorithms stop if the "error" between
                consecutive iterations becomes less than this numer). Default is 1.E-6 but for large graphs
                1.E-9 often yields more robust convergence points.
            error_type: How to calculate the "error" between consecutive iterations of graph signals. Look
                at the class definition's supported error type for details. Default is "mabs".
            max_iters: The number of iterations algorithms can run for. If this number is exceeded,
                an exception is thrown. This could help manage computational resources. Default value is 100,
                and exceeding this value with graph filters often indicates that either graphs have large diameters
                or that algorithms of choice converge particularly slowly.
        """
        self.tol = tol
        self.error_type = error_type.lower()
        self.max_iters = max_iters
        self.min_iters = 0
        self.iteration = 0
        self.last_ranks = None
        self._start_time = None
        self.elapsed_time = None

    def start(self, restart_timer=True):
        """
        Starts the convergence manager

        Args:
            restart_time: If True (default) timing information, such as the number of iterations and wall
                clock time measurement, is reset. Otherwise, this only ensures that convergence manager
                performs one iteration before starting comparing values with previous ones.
        """
        if restart_timer or self._start_time is None:
            self._start_time = time.clock()
            self.elapsed_time = None
            self.iteration = 0
            self.min_iters = 0
        self.last_ranks = None

    def force_next_iteration(self):
        """
        Forcefully performs at least one more iteration.
        """
        self.min_iters = self.iteration+1

    def has_converged(self, new_ranks):
        """
        Checks whether convergence has been by comparing this iteration's numpy array with the previous iteration's.

        Args:
            new_ranks: The iteration's numpy array.
        """
        if self.error_type == "dynamic_iters":
            self._find_max_iters_dynamically(new_ranks)
        self.iteration += 1
        if self.iteration <= self.min_iters:
            self.elapsed_time = time.clock()-self._start_time
            return True
        if self.iteration > self.max_iters:
            if self.error_type=="iters":
                self.elapsed_time = time.clock()-self._start_time
                return True
            raise Exception("Could not converge within "+str(self.max_iters)+" iterations")
        converged = False if self.last_ranks is None else self._has_converged(self.last_ranks, new_ranks)
        self.last_ranks = new_ranks
        self.elapsed_time = time.clock()-self._start_time
        return converged

    def _has_converged(self, prev_ranks, ranks):
        if self.error_type == "iters":
            return False
        if self.error_type == "const":
            return True
        ranks = np.array(ranks)
        if self.error_type == "msqrt":
            return (np.square(ranks - prev_ranks).sum()/ranks.size)**0.5 < self.tol
        elif self.error_type == "mabs":
            return np.absolute(ranks - prev_ranks).sum()/ranks.size < self.tol
        elif self.error_type == "small_value":
            return np.absolute(ranks).sum()/ranks.size < self.tol
        else:
            raise Exception("Supported error types: msqrt, mabs, const, small_value, iters")

    def __str__(self):
        return str(self.iteration)+" iterations ("+str(self.elapsed_time)+" sec)"


class RankOrderConvergenceManager:
    def __init__(self, pagerank_alpha, confidence=0.98, criterion="rank_gap"):
        self.iteration = 0
        self._start_time = None
        self.elapsed_time = None
        self.accumulated_ranks = None
        #self.accumulated_rank_squares = None
        self.pagerank_alpha = pagerank_alpha
        self.confidence = confidence
        self.criterion = criterion
        if self.pagerank_alpha > 1 or self.pagerank_alpha < 0:
            raise Exception("pagerank_alpha must be in the range [0,1] for RankOrderConvergenceManager")

    def start(self, restart_timer=True):
        if restart_timer or self._start_time is None:
            self._start_time = time.clock()
            self.elapsed_time = None
            self.iteration = 0
            #self.accumulated_rank_squares = 0
            self.accumulated_ranks = 0

    def has_converged(self, new_ranks):
        if self._start_time is None:
            raise Exception("Need to start() the convergence manager")
        new_ranks = np.array(new_ranks)
        self.accumulated_ranks = (self.accumulated_ranks*self.iteration + new_ranks) / (self.iteration+1)
        #self.accumulated_rank_squares += (self.accumulated_rank_squares*self.iteration + new_ranks * new_ranks) / (self.iteration+1)
        self.iteration += 1
        converged = self.current_fraction_of_random_walks() >= self.needed_fraction_of_random_walks(new_ranks)
        #print(self.current_fraction_of_random_walks(), self.needed_fraction_of_random_walks(new_ranks))
        self.elapsed_time = time.clock()-self._start_time
        return converged

    def needed_fraction_of_random_walks(self, ranks):
        if self.criterion == "rank_gap":
            a = [rank for rank in ranks]
            order = np.argsort(a, kind='quicksort')
            gaps = [a[order[i + 1]] - a[order[i]] for i in range(len(order)-1) if a[order[i + 1]] != a[order[i]]]
            if len(gaps) < 2:
                return 1
            return 1-(max(gaps)-min(gaps)) / (norm.ppf(self.confidence) * np.std(gaps)*len(gaps))
        elif self.criterion == "fraction_of_walks":
            return self.confidence
        else:
            raise Exception("criterion can only be 'rank_gap' or 'fraction_of_walks'")
        return self.fraction_of_random_walks

    def current_fraction_of_random_walks(self):
        sup_of_series_sum = -np.log(1 - self.pagerank_alpha)
        series_sum = 0
        power = 1
        for n in range(1, self.iteration+1):
            power *= self.pagerank_alpha
            series_sum += power/n # this is faster than np.power(self.pagerank_alpha, n) / n
        return series_sum / sup_of_series_sum

    def __str__(self):
        return str(self.iteration)+" iterations ("+str(self.elapsed_time)+" sec)"