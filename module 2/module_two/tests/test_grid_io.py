import csv
import json
import math
from pathlib import Path
import numpy as np
import pytest
from quality_aware_navigation.grid import GridSpec, load_module_one
from quality_aware_navigation.model import CostModel


FIELDS = ['grid_x', 'grid_y', 'mean_quality', 'variance', 'sample_count', 'last_update', 'known']


def write_grid(tmp_path, rows=None, **metadata_changes):
    meta = dict(artifact_type='localization_quality_grid',
                value_semantics='0_low_quality_1_high_quality', width=2, height=1,
                resolution=.5, origin_x=-2., origin_y=3., coordinate_frame='odom')
    meta.update(metadata_changes)
    (tmp_path / 'quality_grid_metadata.json').write_text(json.dumps(meta), encoding='utf-8')
    path = tmp_path / 'quality_grid.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(rows if rows is not None else [
            [0, 0, .8, .01, 4, 100, 1], [1, 0, 0, 0, 0, 0, 0]])
    return path


def test_load_full_statistics_and_explicit_unknown(tmp_path):
    g = load_module_one(write_grid(tmp_path))
    assert g.spec.frame == 'odom'
    assert g.known.tolist() == [[True, False]]
    assert g.mean[0, 0] == .8
    assert g.count[0, 0] == 4
    assert g.last_observed[0, 0] == 100
    assert CostModel().costs(g, 100).shape == (1, 2)


@pytest.mark.parametrize('rows', [
    [[0, 0, .8, .01, 4, 100, 1]],
    [[0, 0, .8, .01, 4, 100, 1], [0, 0, .8, .01, 4, 100, 1]],
    [[0, 0, .8, .01, 4, 100, 1], [2, 0, 0, 0, 0, 0, 0]],
    [[0, 0, 1.2, .01, 4, 100, 1], [1, 0, 0, 0, 0, 0, 0]],
    [[0, 0, .8, -.01, 4, 100, 1], [1, 0, 0, 0, 0, 0, 0]],
    [[0, 0, .8, .01, 0, 100, 1], [1, 0, 0, 0, 0, 0, 0]],
    [[0, 0, .8, .01, 4, -1, 1], [1, 0, 0, 0, 0, 0, 0]],
    [[0, 0, .8, .01, 4, 100, 2], [1, 0, 0, 0, 0, 0, 0]],
])
def test_corrupt_grid_rejected(tmp_path, rows):
    with pytest.raises(ValueError):
        load_module_one(write_grid(tmp_path, rows))


def test_occupancy_metadata_cannot_be_mistaken_for_quality(tmp_path):
    with pytest.raises(ValueError):
        load_module_one(write_grid(tmp_path, value_semantics='occupancy_probability'))


@pytest.mark.parametrize('yaw', [0, .2, math.pi/2, -2.4])
def test_rotated_grid_cell_round_trip(yaw):
    spec = GridSpec(7, 11, .3, -4.2, 8.7, yaw, 'odom')
    for x in range(spec.width):
        for y in range(spec.height):
            assert spec.world_cell(*spec.cell_center(x, y)) == (x, y)


def test_actual_27_module_one_maps_when_present():
    base = Path(__file__).resolve().parents[2] / 'module_one_reference' / 'localization_quality_final' / 'results' / 'formal_training' / 'quality_maps'
    if not base.is_dir():
        pytest.skip('module-one reference data not distributed with standalone module two')
    paths = sorted(base.glob('*/quality_grid.csv'))
    assert len(paths) == 27
    for path in paths:
        g = load_module_one(path)
        assert g.known.any(), path
        as_of = float(np.max(g.last_observed[g.known]))
        costs = CostModel().costs(g, as_of)
        assert costs.shape == g.spec.shape
        assert np.max(costs) <= 200
