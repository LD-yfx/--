from dataclasses import replace
import numpy as np
import pytest
from quality_aware_navigation.grid import GridSpec, QualityGrid
from quality_aware_navigation.model import CostModel, CostParameters


def cells(n=101, mean=.8, variance=.02, count=20, observed=100):
    g = QualityGrid.empty(GridSpec(n, 1, .1))
    g.known[:] = True
    g.mean[:] = mean
    g.variance[:] = variance
    g.count[:] = count
    g.last_observed[:] = observed
    return g


@pytest.mark.parametrize('mode', CostModel.MODES)
def test_higher_quality_never_costs_more(mode):
    g = cells()
    g.mean[0] = np.linspace(0, 1, 101)
    values = CostModel(mode=mode).costs(g, 100)[0].astype(int)
    assert np.all(np.diff(values) <= 0)


@pytest.mark.parametrize('field,values,direction', [
    ('variance', np.linspace(0, .25, 101), 1),
    ('count', np.arange(1, 102), -1),
    ('last_observed', np.linspace(0, 100, 101), -1),
])
def test_evidence_and_freshness_monotonicity(field, values, direction):
    g = cells()
    getattr(g, field)[0] = values
    risks = CostModel().risks(g, 100)[0]
    assert np.all(direction * np.diff(risks) >= -1e-12)


def test_aging_never_rehabilitates_a_known_bad_cell():
    g = cells(1, mean=.05, variance=0, count=10000)
    model = CostModel()
    assert model.risks(g, 100)[0, 0] == pytest.approx(.95)
    assert model.risks(g, 100000)[0, 0] == pytest.approx(.95)


def test_sparse_or_old_good_cell_approaches_prior_from_below():
    g = cells(1, mean=1, variance=0, count=10000)
    model = CostModel()
    assert model.risks(g, 100)[0, 0] == pytest.approx(0)
    assert model.risks(g, 100000)[0, 0] == pytest.approx(.7)
    g.count[:] = 1
    assert 0 < model.risks(g, 100)[0, 0] < .7


def test_unknown_is_explicit_and_does_not_consume_observation_placeholders():
    g = QualityGrid.empty(GridSpec(2, 2, .1))
    g.mean[:] = np.nan
    g.variance[:] = np.nan
    g.count[:] = np.nan
    g.last_observed[:] = np.nan
    assert np.all(CostModel().risks(g, 100) == .7)


def test_mean_only_does_not_invent_statistics():
    g = cells(2, mean=.99)
    g.statistics_mode = 'mean_only'
    g.variance[:] = np.nan
    g.count[:] = np.nan
    g.last_observed[:] = np.nan
    assert np.all(CostModel().risks(g, 100) == .7)
    g.mean[:] = .05
    assert np.allclose(CostModel().risks(g, 100), .95)


@pytest.mark.parametrize('mode', CostModel.MODES)
def test_costs_do_not_alias_reserved_obstacle_values(mode):
    rng = np.random.default_rng(73)
    g = cells(1000)
    g.mean[:] = rng.random(g.spec.shape)
    g.variance[:] = rng.random(g.spec.shape)
    g.count[:] = rng.integers(1, 100, g.spec.shape)
    g.last_observed[:] = rng.uniform(0, 100, g.spec.shape)
    out = CostModel(CostParameters(max_cost=252), mode).costs(g, 100)
    assert out.dtype == np.uint8
    assert out.max() <= 252


@pytest.mark.parametrize('field,value', [
    ('beta', -1), ('gamma', 0), ('sample_scale', 0), ('age_scale', -1),
    ('unknown_risk', 1.1), ('max_cost', 253), ('max_cost', 2.5),
    ('mean_only_floor', -.1), ('beta', float('nan')),
])
def test_invalid_model_parameters_rejected(field, value):
    with pytest.raises(ValueError):
        replace(CostParameters(), **{field: value})


def test_source_clock_mismatch_is_rejected_for_statistics_model():
    g = cells(1, observed=1234)
    with pytest.raises(ValueError, match='future'):
        CostModel().risks(g, 100)
