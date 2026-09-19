"""Check synthetic inputs are reproducible and statistically realizable."""
import numpy as np
import pytest
from run_experiments import make_scene, SCENARIOS


@pytest.mark.parametrize('name', SCENARIOS)
def test_same_seed_gives_identical_scene_and_feasible_bounded_moments(name):
    a, b = make_scene(name, 100), make_scene(name, 100)
    assert np.array_equal(a.occupied, b.occupied)
    assert np.array_equal(a.quality.mean, b.quality.mean)
    assert np.array_equal(a.latent_quality, b.latent_quality)
    g = a.quality
    one = g.known & (g.count == 1)
    assert np.all(g.variance[one] == 0)
    multiple = g.known & (g.count > 1)
    mu, n = g.mean[multiple], g.count[multiple]
    # Any set of samples in [0,1] must obey this unbiased-variance bound.
    assert np.all(g.variance[multiple] <= mu * (1-mu) * n / (n-1) + 1e-12)


def test_stress_scenarios_use_same_collision_geometry_and_exogenous_diagnostic():
    baseline = make_scene('dual_route', 103)
    for name in SCENARIOS:
        scene = make_scene(name, 103)
        if name != 'no_path':
            assert np.array_equal(scene.occupied, baseline.occupied)
        assert scene.start == baseline.start and scene.goal == baseline.goal
        assert np.array_equal(scene.latent_quality, baseline.latent_quality)


def test_distinct_seeds_change_observations_without_changing_geometry():
    a, b = make_scene('dual_route', 100), make_scene('dual_route', 101)
    assert np.array_equal(a.occupied, b.occupied)
    assert not np.array_equal(a.quality.mean, b.quality.mean)
