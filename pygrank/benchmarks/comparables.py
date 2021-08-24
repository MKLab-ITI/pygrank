from typing import Callable, Mapping
from pygrank.algorithms import NodeRanking, preprocessor, PageRank, AbsorbingWalks, HeatKernel, BiasedKernel
from pygrank.algorithms import Postprocessor, Tautology, Sweep, SeedOversampling


def create_demo_filters(pre=None,
                        tol: float = 1.E-9,
                        max_iters: int = 1000) -> Mapping[str, NodeRanking]:
    """
    Creates six commonly used graph filters with automatic normalization and immutability. These are applicable
    to a wide range of real-world scenarios. Filters share the same preprocessor to run experiments faster.

    Args:
        pre: A preprocessor to use for generated methods. If None (default) a preprocessor with assumed immutability
            and automatic normalization will be used.
        tol: A tolerance parameter to control convergence points of filters. Default is 1.E-9.
        max_iters: The maximum iterations filters should run for. Should be large to accommodate slow convergence.
            Default is 1000.

    Returns:
        A map from filter names to algorithms.
    """
    if pre is None:
        pre = preprocessor(assume_immutability=True)
    return {"PPR.85": PageRank(alpha=0.85, preprocessor=pre, max_iters=max_iters, tol=tol),
            "PPR.9": PageRank(alpha=0.95, preprocessor=pre, max_iters=max_iters, tol=tol),
            "PPR.99": PageRank(alpha=0.95, preprocessor=pre, max_iters=max_iters, tol=tol),
            "HK3": HeatKernel(t=3, preprocessor=pre, max_iters=max_iters, tol=tol),
            "HK5": HeatKernel(t=5, preprocessor=pre, max_iters=max_iters, tol=tol),
            "HK7": HeatKernel(t=7, preprocessor=pre, max_iters=max_iters, tol=tol)
            }


def create_many_filters(tol: float = 1.E-6,
                        max_iters: int = 10000) -> Mapping[str, NodeRanking]:
    """
    Creates a wide range of base filters to use for benchmarking of evaluation measures. It is recommended that
    variations of recommended filters are also applied. This includes both symmetric and non-symmetric filters.
    Filters share the same preprocessor to run experiments faster.

    Args:
        tol: A tolerance parameter to control convergence points of filters. Default is 1.E-6.
        max_iters: The maximum iterations filters should run for. Should be large to accommodate slow convergence.
            Default is 1000.

    Returns:
        A map from filter names to algorithms.
    """
    pre = preprocessor('col', assume_immutability=True)
    preL = preprocessor('symmetric', assume_immutability=True)
    return {
        "PPRL.85": PageRank(alpha=0.85, preprocessor=preL, max_iters=max_iters, tol=tol),
        "PPRL.90": PageRank(alpha=0.9, preprocessor=preL, max_iters=max_iters, tol=tol),
        "PPRL.95": PageRank(alpha=0.95, preprocessor=preL, max_iters=max_iters, tol=tol),
        "PPRL.99": PageRank(alpha=0.99, preprocessor=preL, max_iters=max_iters, tol=tol),
        "PPR.85": PageRank(alpha=0.85, preprocessor=pre, max_iters=max_iters, tol=tol),
        "PPR.90": PageRank(alpha=0.9, preprocessor=pre, max_iters=max_iters, tol=tol),
        "PPR.95": PageRank(alpha=0.95, preprocessor=pre, max_iters=max_iters, tol=tol),
        "PPR.99": PageRank(alpha=0.99, preprocessor=pre, max_iters=max_iters, tol=tol),
        "HK1": HeatKernel(t=1, preprocessor=pre, max_iters=max_iters, tol=tol),
        "HK3": HeatKernel(t=3, preprocessor=pre, max_iters=max_iters, tol=tol),
        "HK5": HeatKernel(t=5, preprocessor=pre, max_iters=max_iters, tol=tol),
        "HK7": HeatKernel(t=7, preprocessor=pre, max_iters=max_iters, tol=tol),
        "HKL1": HeatKernel(t=1, preprocessor=preL, max_iters=max_iters, tol=tol),
        "HKL3": HeatKernel(t=3, preprocessor=preL, max_iters=max_iters, tol=tol),
        "HKL5": HeatKernel(t=5, preprocessor=preL, max_iters=max_iters, tol=tol),
        "HKL7": HeatKernel(t=7, preprocessor=preL, max_iters=max_iters, tol=tol),
        "AbsorbL.85": AbsorbingWalks(alpha=0.85, preprocessor=preL, max_iters=max_iters, tol=tol),
        "AbsorbL.90": AbsorbingWalks(alpha=0.9, preprocessor=preL, max_iters=max_iters, tol=tol),
        "AbsorbL.95": AbsorbingWalks(alpha=0.95, preprocessor=preL, max_iters=max_iters, tol=tol),
        "AbsorbL.99": AbsorbingWalks(alpha=0.99, preprocessor=preL, max_iters=max_iters, tol=tol),
        "Absorb.85": AbsorbingWalks(alpha=0.85, preprocessor=pre, max_iters=max_iters, tol=tol),
        "Absorb.90": AbsorbingWalks(alpha=0.9, preprocessor=pre, max_iters=max_iters, tol=tol),
        "Absorb.95": AbsorbingWalks(alpha=0.95, preprocessor=pre, max_iters=max_iters, tol=tol),
        "Absorb.99": AbsorbingWalks(alpha=0.99, preprocessor=pre, max_iters=max_iters, tol=tol),
    }


def create_many_variation_types() -> Mapping[str, Callable[[NodeRanking], Postprocessor]]:
    return {"": Tautology,
            "Sweep": Sweep,
            "O": lambda alg: SeedOversampling(alg, "safe"),
            "SweepO": lambda alg: SeedOversampling(Sweep(alg), "safe"),
            "Neigh": lambda alg: SeedOversampling(alg, "neighbors"),
            "SweepNeigh": lambda alg: SeedOversampling(Sweep(alg), "neighbors")
            }


def create_variations(algorithms: Mapping[str, NodeRanking],
                      variations: Mapping[str, Callable[[NodeRanking], Postprocessor]]):
    """
    Augments provided algorithms with all possible variations.
    Args:
        algorithms: A map from names to node ranking algorithms to compare.
        variations: A map from names to postprocessor types to wrap around node ranking algorithms.
    Returns:
        A map from names to node ranking algorithms to compare. New names append the variation name.
    Example:
        >>> import pygrank as pg
        >>> algorithms = pg.create_variations(pg.create_many_filters(), pg.create_many_variation_types())
    """
    all_algorithms = dict()
    for variation in variations:
        for algorithm in algorithms:
            all_algorithms[algorithm + variation] = variations[variation](algorithms[algorithm])
    return all_algorithms
