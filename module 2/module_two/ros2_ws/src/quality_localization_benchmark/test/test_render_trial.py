"""Visualization semantics: do not turn gaps or resets into observed trajectories."""
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quality_localization_benchmark.render_trial import (
    split_segments, missing_intervals, transform_rows, fixed_transform,
    mission_window, render_trial,
)


def test_temporal_gaps_invalid_rows_reversals_and_teleports_break_lines():
    rows = [
        dict(t=0., x=0., y=0.), dict(t=.05, x=.01, y=0.),
        dict(t=.4, x=.03, y=0.), dict(t=.45, x=float('nan'), y=0.),
        dict(t=.5, x=.04, y=0.), dict(t=.4, x=.05, y=0.),
        dict(t=.45, x=10., y=0.), dict(t=.5, x=10.01, y=0.),
    ]
    segments = split_segments(rows, max_gap=.1, max_step=.5)
    assert [len(s) for s in segments] == [2, 1, 1, 1, 2]
    assert all(0 < b['t']-a['t'] <= .1 for seg in segments for a, b in zip(seg, seg[1:]))
    assert all(math.dist((a['x'], a['y']), (b['x'], b['y'])) <= .5
               for seg in segments for a, b in zip(seg, seg[1:]))


def test_explicit_resets_and_segment_markers_are_not_joined():
    rows = [dict(t=0., x=0., y=0., reset_id=1),
            dict(t=.05, x=.01, y=0., reset_id=2),
            dict(t=.1, x=.02, y=0., reset=True)]
    assert [len(s) for s in split_segments(rows)] == [1, 1, 1]


def test_missing_quality_value_breaks_curve_and_remains_missing():
    rows = [dict(t=0., q_fused=.8), dict(t=.1, q_fused=None), dict(t=.2, q_fused=.9)]
    segments = split_segments(rows, ('q_fused',), max_gap=.5)
    assert [len(s) for s in segments] == [1, 1]
    assert missing_intervals(rows, ('q_fused',), (0., .2), .02) == [[.02, .18000000000000002]]
    assert missing_intervals([], ('x', 'y'), (0., 1.)) == [[0., 1.]]


def test_fixed_frame_transform_is_applied_without_data_fit():
    transform = fixed_transform({'frame_transform': dict(x=3., y=-2., yaw=math.pi/2)})
    output = transform_rows([dict(t=0., x=1., y=0.)], transform)
    assert output[0]['x'] == pytest.approx(3.)
    assert output[0]['y'] == pytest.approx(-1.)
    with pytest.raises(ValueError, match='finite'):
        fixed_transform({'frame_transform': {'x': float('nan')}})


def test_mission_window_prefers_boundaries_and_labels_recording_fallback():
    actions = {'leg_actions': [dict(mission_start=10., mission_end=20.),
                               dict(mission_start=21., mission_end=30.)]}
    window, scope = mission_window({}, actions, [[dict(t=0.), dict(t=40.)]])
    assert window == (10., 30.) and scope == 'recorded_action_envelope'
    window, scope = mission_window({}, {}, [[dict(t=0.), dict(t=40.)]])
    assert window == (0., 40.) and 'recording_window_only' in scope


def test_trial_renderer_records_gaps_and_missing_inputs_without_overwriting(tmp_path):
    pytest.importorskip('matplotlib')
    trial = tmp_path/'trial'
    trial.mkdir()
    manifest = dict(run_id='fixture_only', map_id='fixture', method='geometric',
                    mission_start=0., mission_end=1., planned_waypoints=[[1., 0.]],
                    frame_transform=dict(x=0., y=0., yaw=0.),
                    metadata=dict(rectangles=[[-1., -.8, -1., 1.]], bounds=[-1., 2., -1., 1.]))
    (trial/'manifest.json').write_text(json.dumps(manifest))
    (trial/'result.json').write_text(json.dumps(dict(completed=True, valid_evidence=True,
                                                    pose_metrics_valid=False, success=False)))
    truth = [dict(t=i/20., x=i/20., y=0.) for i in range(21)]
    estimates = [dict(row, x=row['x']+.1) for row in truth if row['t'] <= .2 or row['t'] >= .8]
    quality = [dict(t=row['t'], q_fused=.8, q_visual=None, q_laser=.9,
                    error_position_m=.1 if row['t'] <= .2 else None) for row in truth]
    for name, rows in [('truth', truth), ('estimates', estimates), ('quality_pairs', quality)]:
        (trial/(name+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    output = render_trial(trial, tmp_path/'figures')
    assert (output/'trajectory.png').read_bytes().startswith(b'\x89PNG')
    assert (output/'error_quality_timeline.png').read_bytes().startswith(b'\x89PNG')
    audit = json.loads((output/'render_manifest.json').read_text())
    assert audit['trajectory_segments']['estimates'] == 2
    assert audit['missing_intervals']['q_visual'] == [[0., 1.]]
    assert audit['missing_intervals']['estimate'][0][0] == pytest.approx(.25)
    assert audit['missing_intervals']['estimate'][0][1] == pytest.approx(.75)
    assert not audit['input_files']['plans.jsonl']['exists']
    assert 'lower-rate' in audit['error_source']
    with pytest.raises(FileExistsError):
        render_trial(trial, tmp_path/'figures')
