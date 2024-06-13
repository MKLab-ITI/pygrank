import numpy as np
from scipy.stats import norm
from timeit import default_timer as time
from pygrank.measures import Supervised, Mabs
from pygrank.core import backend, BackendPrimitive
from typing import Union
import warnings


class ConvergenceManager:
    """Used to keep previous iteration and generally manage convergence of variables. Graph filters
    automatically create instances of this class by passing on appropriate parameters.

    Examples:
        >>> convergence = ConvergenceManager()
        >>> convergence.start()
        >>> var = None
        >>> while not convergence.has_converged(var):
        >>>     ...
        >>>     var = ...
    """

    def __init__(
        self,
        tol: float = 1.0e-6,
        error_type: Union[Supervised, str] = Mabs,
        max_iters: int = 100,
        end_modulo: int = 1,
        iter_exception=Exception,
    ):
        """
        Initializes a convergence manager with a provided tolerance level, error type and number of iterations.

        Args:
            tol: Numerical tolerance to determine the stopping point (algorithms stop if the "error" between
                consecutive iterations becomes less than this number). Default is 1.E-6 but for large graphs
                1.E-9 often yields more robust convergence points. If the provided value is less than the
                numerical precision of the backend `pygrank.epsilon()` then it is snapped to that value.
                *None* tolerance will stop when consecutive iterations are exactly the same.
            error_type: Optional. How to calculate the "error" between consecutive iterations of graph signals.
                If "iters", convergence is reached at iteration *max_iters*-1 without throwing an exception
                and even if numerical convergence happens to occur earlier. Default is `pygrank.Mabs`.
            max_iters: Optional. The number of iterations algorithms can run for. If this number is exceeded,
                an exception is thrown. This could help manage computational resources. Default value is 100,
                and exceeding this value with graph filters often indicates that either graphs have large diameters
                or that algorithms of choice converge particularly slowly.
            end_modulo. Optional. Checks the convergence criteria every fixed number of iterations. For value of
                1 (default), convergence is checked in every iteration, for value of 2 every second iteration, etc.
            iter_exception: Optional. The type of exception class to be thrown if max iterations are reached (when
                *error_type* is not "iters"). If *None*, this quietly closes the iterations as if convergence
                is reached. *Avoid* changing this argument for deployment-ready systems, as performing a fixed number
                of iterations should be a preferred practice compared to stopping either there or at a fixed numerical
                tolerance. Default is *Exception*.
        """
        self.tol = tol
        self.error_type = error_type
        self.max_iters = max_iters
        self.iteration = 0
        self.last_ranks = None
        self._start_time = None
        self.elapsed_time = None
        self.iter_exception = iter_exception
        self.end_modulo = end_modulo

    def start(self, restart_timer: bool = True):
        """
        Starts the convergence manager

        Args:
            restart_timer: Optional. If True (default) timing information, such as the number of iterations and wall
                clock time measurement, is reset. Otherwise, this only ensures that the convergence manager
                performs one iteration before starting comparing values with previous ones.
        """
        if restart_timer or self._start_time is None:
            self._start_time = time()
            self.elapsed_time = None
            self.iteration = 0
        self.last_ranks = None

    def has_converged(self, new_ranks: BackendPrimitive) -> bool:
        """
        Checks whether convergence has been achieved by comparing this iteration's backend array with the
        previous iteration's.

        Args:
            new_ranks: The iteration's backend array.
        """
        self.iteration += 1
        if self.iteration >= self.max_iters:
            if self.error_type == "iters" or self.iter_exception is None:
                self.elapsed_time = time() - self._start_time
                return True
            raise self.iter_exception(
                "Could not converge within " + str(self.max_iters) + " iterations"
            )
        converged = (
            False
            if self.last_ranks is None
            else self._has_converged(self.last_ranks, new_ranks)
        )
        self.last_ranks = new_ranks
        if converged:
            self.elapsed_time = time() - self._start_time
        return converged

    def _has_converged(
        self, prev_ranks: BackendPrimitive, ranks: BackendPrimitive
    ) -> bool:
        if self.error_type == "iters":
            return False
        if self.iteration % self.end_modulo != 0:
            return False
        err = self.error_type(prev_ranks)
        tol = 0 if self.tol is None else max(self.tol, backend.epsilon())
        if err.best_direction() <= 0:
            return err(ranks) <= tol
        return err(ranks) >= 1 - tol

    def __str__(self):
        return str(self.iteration) + " iterations (" + str(self.elapsed_time) + " sec)"


class RankOrderConvergenceManager:
    def __init__(
        self,
        pagerank_alpha: float,
        confidence: float = 0.98,
        criterion: str = "rank_gap",
    ):
        # TODO: add documentation
        self.iteration = 0
        self._start_time = None
        self.elapsed_time = None
        self._accumulated_ranks = None
        self.pagerank_alpha = pagerank_alpha
        self.confidence = confidence
        self.criterion = criterion
        self._power = 1
        self._sup_of_series_sum = -np.log(1 - self.pagerank_alpha)
        self._series_sum = 0
        self._warned = False
        self._conf_ppf = norm.ppf(self.confidence)
        self._targeting_fraction = 0
        if isinstance(criterion, int):
            if criterion<4:
                raise Exception(f"When initializing `clever_gap` with the number of node samples, "
                                f"provide at least 2 node samples instead of {criterion}")
        elif criterion not in ["fraction_of_walks", "clever_gap", "rank_gap"]:
            raise Exception(f"RankOrderConvergenceManager got argument criterion='{criterion}'"
                            f"but one of 'fraction_of_walks', 'clever_gap', 'rank_gap' or an int"
                            f"should be provided")

    def start(self, restart_timer: bool = True):
        if restart_timer or self._start_time is None:
            self._start_time = time()
            self.elapsed_time = None
            self.iteration = 0
            self._accumulated_ranks = 0
            self._sup_of_series_sum = -np.log(1 - self.pagerank_alpha)
            self._series_sum = 0
            self._power = 1
            self._warned = False
            self._conf_ppf = norm.ppf(self.confidence)
            self._targeting_fraction = 0

    def has_converged(self, new_ranks: BackendPrimitive) -> bool:

        self.iteration += 1
        self._power *= self.pagerank_alpha
        self._series_sum += self._power / self.iteration

        current_fraction_random_walks = self.current_fraction_of_random_walks()

        if self.iteration < 5:
            converged = False
        elif (
            current_fraction_random_walks >= self._targeting_fraction
            or self.iteration % 10 == 0
        ):
            new_ranks = backend.to_numpy(new_ranks)
            # TODO: convert to any backend
            self._targeting_fraction = self.needed_fraction_of_random_walks(new_ranks)
            converged = current_fraction_random_walks >= self._targeting_fraction
        else:
            converged = False
        if converged:
            self.elapsed_time = time() - self._start_time
        return converged

    def needed_fraction_of_random_walks(self, ranks: BackendPrimitive) -> float:
        criterion = self.criterion
        if backend.length(ranks) < 30 and (
            isinstance(self.criterion, int) or self.criterion != "fraction_of_walks"
        ):
            if not self._warned:
                self._warned = True
                warnings.warn(
                    f"Inapplicable convergence criterion."
                    f"\n  Description: "
                    f"\n    RankOrderConvergenceManager was initialized with criterion='{self.criterion}'"
                    f"\n    but this is applicable only if the central limit theorem "
                    f"\n    can assume normal distribution for a random variable over {len(ranks)} nodes."
                    f"\n  How to avoid this warning:"
                    f"\n    Supply a graph with at least 30 nodes or use criterion='fraction_of_walks'."
                    f"\n  Temporary fix:"
                    f"\n    Using criterion='fraction_of_walks' for this round of convergence."
                )
            criterion = "fraction_of_walks"
        if isinstance(criterion, int) or criterion == "clever_gap":
            n_gap = (
                self.criterion
                if isinstance(self.criterion, int)
                else int(backend.length(ranks) ** 0.5)
            )
            a = np.random.choice(ranks, n_gap)
            a.sort()
            gaps = np.diff(a)
            gaps = gaps[gaps != 0]
            if len(gaps) < 2:
                return 1
            return 1 - np.quantile(gaps, 1 - self.confidence) / (
                self._conf_ppf * np.std(gaps)
            )
        elif criterion == "rank_gap":
            """a = ranks
            order = np.sort(a, kind="quicksort")
            gaps = np.diff(order)
            gaps = gaps[gaps != 0]
            if len(gaps) < 2:
                return 1
            return 1 - (np.max(a) - np.min(a)) / (
                norm.ppf(self.confidence) * np.std(gaps) * len(gaps)
            )"""
            a = np.sort(ranks, kind="quicksort")
            gaps = np.diff(a)
            gaps = gaps[gaps != 0]
            if len(gaps) < 2:
                return 1
            return 1 - np.quantile(gaps, 1 - self.confidence) / (
                self._conf_ppf * np.std(gaps)
            )
        elif criterion == "fraction_of_walks":
            return self.confidence
        else:
            raise Exception("criterion can only be 'rank_gap' or 'fraction_of_walks'")

    def current_fraction_of_random_walks(self) -> float:
        return self._series_sum / self._sup_of_series_sum

    def __str__(self):
        return str(self.iteration) + " iterations (" + str(self.elapsed_time) + " sec)"
