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


def test_autotune_citations():
    assert pg.ParameterTuner().cite() != pg.GenericGraphFilter().cite()
    assert pg.HopTuner().cite() != pg.GenericGraphFilter().cite()
    assert pg.AlgorithmSelection().cite() != pg.GenericGraphFilter().cite()


def test_filter_citations():
    assert pg.PageRank().cite() != pg.GraphFilter().cite()
    assert pg.HeatKernel().cite() != pg.GraphFilter().cite()
    assert pg.AbsorbingWalks().cite() != pg.GraphFilter().cite()
    assert pg.HeatKernel().cite() != pg.GraphFilter().cite()
    assert pg.PageRank(alpha=0.85).cite() != pg.PageRank(alpha=0.99).cite()
    assert pg.HeatKernel(krylov_dims=0).cite() != pg.HeatKernel(krylov_dims=5).cite()
    assert pg.HeatKernel(coefficient_type="taylor").cite() != pg.HeatKernel(coefficient_type="chebyshev").cite()
    assert pg.HeatKernel(optimization_dict=dict()).cite() != pg.HeatKernel(optimization_dict=None).cite()


def test_postprocessor_citations():
    assert pg.Tautology(pg.PageRank()).cite() == pg.PageRank().cite()
    assert pg.Normalize(pg.PageRank()).cite() != pg.PageRank().cite()
    assert pg.Normalize(pg.PageRank(), "sum").cite() != pg.Normalize(pg.PageRank(), "range").cite()
    assert pg.Ordinals(pg.PageRank()).cite() != pg.Normalize(pg.PageRank(), "sum").cite()
    assert pg.Transformer(pg.PageRank()).cite() != pg.PageRank().cite()
    assert pg.Threshold(pg.PageRank()).cite() != pg.PageRank().cite()
    assert pg.Sweep(pg.PageRank()).cite() != pg.PageRank().cite()
    assert pg.BoostedSeedOversampling(pg.PageRank()).cite() != pg.PageRank().cite()
    assert pg.SeedOversampling(pg.PageRank()).cite() != pg.PageRank().cite()
    assert pg.SeedOversampling(pg.PageRank(), method="safe").cite() \
           != pg.SeedOversampling(pg.PageRank(), method="top").cite()
    assert pg.BoostedSeedOversampling(pg.PageRank(), objective="partial").cite() \
           != pg.BoostedSeedOversampling(pg.PageRank(), objective="naive").cite()
    assert pg.BoostedSeedOversampling(pg.PageRank(), oversample_from_iteration="previous").cite() \
           != pg.BoostedSeedOversampling(pg.PageRank(), oversample_from_iteration="original").cite()
    # TODO: add fairness citation tests
