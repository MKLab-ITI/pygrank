import pygrank as pg
import pytest


def test_autorefs():
    """
    Tests that different (base) algorithms yield different citations, that all citations have at least one
    reference to a publication and that wrapping the same base algorithms yields the same citations.
    """
    pre = pg.preprocessor(assume_immutability=True, normalization="symmetric")
    algs = {"ppr.85": pg.PageRank(.85, preprocessor=pre, tol=1.E-9, max_iters=1000),
            "ppr.99": pg.PageRank(.99, preprocessor=pre, tol=1.E-9, max_iters=1000),
            "hk3": pg.HeatKernel(3, preprocessor=pre, tol=1.E-9, max_iters=1000),
            "hk5": pg.HeatKernel(5, preprocessor=pre, tol=1.E-9, max_iters=1000),
            "hk5'": pg.HeatKernel(5, preprocessor=pre, tol=1.E-9, max_iters=1000),
            }
    algs = algs | pg.create_variations(algs, {"+Sweep": pg.Sweep, "+SO": pg.SeedOversampling, "+BSO": pg.BoostedSeedOversampling})
    citations = set()
    for alg in algs.values():
        citation = alg.cite()
        assert "\\cite{" in citation
        citations.add(citation)
    assert len(citations) == len(algs)-4
