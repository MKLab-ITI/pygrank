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


def test_explicit_citations():
    assert "unknown node ranking algorithm" == pg.NodeRanking().cite()
    assert "with parameters tuned \\cite{krasanakis2022autogf}" in pg.ParameterTuner(
        lambda params: pg.PageRank(params[0])).cite()
    assert "Postprocessor" in pg.Postprocessor().cite()
    assert pg.PageRank().cite() in pg.AlgorithmSelection().cite()
    assert "krasanakis2022autogf" in pg.ParameterTuner().cite()
    assert "ortega2018graph" in pg.ParameterTuner().cite()
    assert pg.HeatKernel().cite() in pg.SeedOversampling(pg.HeatKernel()).cite()
    assert "2 " in pg.Top(2).cite()
    assert ".500 " in pg.Top(.5).cite()
    assert pg.AbsorbingWalks().cite() in pg.BoostedSeedOversampling(pg.AbsorbingWalks()).cite()
    assert "krasanakis2018venuerank" in pg.BiasedKernel(converge_to_eigenvectors=True).cite()
    assert "yu2021chebyshev" in pg.HeatKernel(coefficient_type="chebyshev").cite()
    assert "susnjara2015accelerated" in pg.HeatKernel(krylov_dims=5).cite()
    assert "krasanakis2021pygrank" in pg.GenericGraphFilter(optimization_dict=dict()).cite()
    assert "tautology" in pg.Tautology().cite()
    assert pg.PageRank().cite() == pg.Tautology(pg.PageRank()).cite()
    assert "mabs" in pg.MabsMaintain(pg.PageRank()).cite()
    assert "max normalization" in pg.Normalize(pg.PageRank()).cite()
    assert "[0,1] range" in pg.Normalize(pg.PageRank(), "range").cite()
    assert "ordinal" in pg.Ordinals(pg.PageRank()).cite()
    assert "exp" in pg.Transformer(pg.PageRank()).cite()
    assert "0.5" in pg.Threshold(pg.PageRank(), 0.5).cite()
    assert "andersen2007local" in pg.Sweep(pg.PageRank()).cite()
    assert pg.HeatKernel().cite() in pg.Sweep(pg.PageRank(), pg.HeatKernel()).cite()
    assert "LFPRO" in pg.AdHocFairness("O").cite()
    assert "LFPRO" in pg.AdHocFairness(pg.PageRank(), "LFPRO").cite()
    assert "multiplicative" in pg.AdHocFairness(pg.PageRank(), "B").cite()
    assert "multiplicative" in pg.AdHocFairness(pg.PageRank(), "mult").cite()
    assert "tsioutsiouliklis2020fairness" in pg.AdHocFairness().cite()
    assert "rahman2019fairwalk" in pg.FairWalk(pg.PageRank()).cite()
    assert "krasanakis2020prioredit" in pg.FairPersonalizer(pg.PageRank()).cite()
    assert "tsioutsiouliklis2021fairness" in pg.LFPR().cite()
    assert "uniform" in pg.LFPR().cite()


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
