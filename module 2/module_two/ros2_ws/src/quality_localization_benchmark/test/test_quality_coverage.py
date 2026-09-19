"""Causal sparse-map coverage metrics, with explicit unknown/missing evidence."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quality_localization_benchmark.quality_coverage import summarize_quality_coverage


METADATA={'origin':[0.,0.],'rectangles':[],'world_to_map':[0.,0.,0.],
          'robot_collision_radius':.32}


def snapshot(t=0.,indices=(0,),means=None,received=None):
    return dict(t=t,received_ros=t if received is None else received,frame='map',mode='windowed',
                width=2,height=1,resolution=1.,origin=[0.,0.,0.],indices=list(indices),
                mean=list(means) if means is not None else [.8]*len(indices),
                variance=[.5]*len(indices),sample_count=[2]*len(indices),
                last_observed=[t]*len(indices))


def summarize(stats,plans=(),estimates=(),truth=(),metadata=None,start=0.,end=1.):
    return summarize_quality_coverage(stats,plans,estimates,truth,
                                      METADATA if metadata is None else metadata,start,end)


class QualityCoverageTest(unittest.TestCase):
    def test_unknown_cells_are_excluded_and_known_zero_quality_is_retained(self):
        result=summarize([snapshot(means=[0.])])
        self.assertEqual(result['snapshot_rows'][0]['known_cells'],1)
        self.assertEqual(result['snapshot_rows'][0]['unknown_cells'],1)
        self.assertEqual(result['known_cell_snapshot_distributions']['quality_mean']['n'],1)
        self.assertEqual(result['known_cell_snapshot_distributions']['quality_mean']['median'],0)
        self.assertEqual(result['known_cell_snapshot_distributions']['variance']['maximum'],.5)

    def test_empty_snapshot_is_valid_unknown_not_missing_or_zero_quality(self):
        result=summarize([snapshot(indices=[])])
        self.assertEqual(result['status'],'available')
        self.assertEqual(result['snapshot_rows'][0]['known_fraction'],0)
        self.assertEqual(result['known_cell_snapshot_distributions']['quality_mean']['n'],0)

    def test_plan_is_not_given_snapshot_before_recorder_received_it(self):
        plans=[dict(t=.1,received_ros=.1,points=[[.25,.5],[1.75,.5]]),
               dict(t=.6,received_ros=.6,points=[[.25,.5],[1.75,.5]])]
        result=summarize([snapshot(t=0.,received=.5)],plans)
        before,after=result['plans']['rows']
        self.assertAlmostEqual(before['length_m_by_status']['no_snapshot'],1.5)
        self.assertAlmostEqual(after['known_length_fraction'],.5)
        self.assertEqual(result['availability_time'],'max_source_and_recorder_receipt')

    def test_future_snapshot_is_not_used_even_when_receipt_time_is_earlier(self):
        result=summarize([snapshot(t=.8,received=.2)],
                         [dict(t=.5,points=[[0,.5],[2,.5]])])
        self.assertEqual(result['plans']['rows'][0]['known_length_fraction'],0)
        self.assertEqual(result['plans']['rows'][0]['length_m_by_status']['no_snapshot'],2)

    def test_expiration_removes_known_evidence_without_inventing_bad_quality(self):
        result=summarize([snapshot(0.,means=[.1]),snapshot(.5,indices=[])])
        self.assertEqual(result['transitions']['known_to_unknown'],1)
        self.assertEqual(result['transitions']['removed_known_mean']['median'],.1)
        self.assertEqual(result['known_cell_snapshot_distributions']['quality_mean']['n'],1)

    def test_geometry_change_is_not_counted_as_evidence_expiration(self):
        changed=snapshot(.5,indices=[1]);changed['origin']=[5.,0.,0.]
        result=summarize([snapshot(),changed])
        self.assertEqual(result['transitions']['geometry_changes'],1)
        self.assertEqual(result['transitions']['known_to_unknown'],0)

    def test_pose_gaps_are_not_filled_to_claim_complete_known_time(self):
        result=summarize([snapshot()],estimates=[dict(t=.5,x=.5,y=.5)])
        trajectory=result['estimated_trajectory']
        self.assertAlmostEqual(trajectory['known_seconds'],.1)
        self.assertAlmostEqual(trajectory['missing_pose_seconds'],.9)
        self.assertAlmostEqual(trajectory['known_fraction_of_mission_time'],.1)
        self.assertAlmostEqual(trajectory['known_fraction_of_observed_pose_time'],1.)

    def test_only_frozen_world_to_map_transform_is_used_for_truth(self):
        metadata=copy.deepcopy(METADATA);metadata['world_to_map']=[-5.,0.,0.]
        result=summarize([snapshot()],truth=[dict(t=.5,x=5.5,y=.5,frame='world')],metadata=metadata)
        self.assertAlmostEqual(result['truth_trajectory']['known_seconds'],.1)

    def test_malformed_snapshot_is_an_invalid_barrier_not_silent_old_map_reuse(self):
        malformed=snapshot(.5);malformed['indices']=[0,0]
        result=summarize([snapshot(),malformed],estimates=[dict(t=.75,x=.5,y=.5)])
        self.assertEqual(result['invalid_snapshot_count'],1)
        self.assertAlmostEqual(result['estimated_trajectory']['seconds_by_status']['invalid_snapshot'],.1)

    def test_footprint_safe_coverage_uses_exact_geometry_and_separate_denominator(self):
        metadata=copy.deepcopy(METADATA);metadata['rectangles']=[[0.,.9,0.,1.]]
        result=summarize([snapshot()],metadata=metadata)
        row=result['snapshot_rows'][0]
        self.assertEqual(row['footprint_safe_cell_centres'],1)
        self.assertEqual(row['known_footprint_safe_cells'],0)
        self.assertEqual(row['known_fraction'],.5)
        self.assertEqual(row['known_fraction_of_footprint_safe_cells'],0)

    def test_plan_length_is_invariant_to_point_density_and_handles_outside_map(self):
        plans=[dict(t=.5,points=[[-1.,.5],[3.,.5]]),
               dict(t=.5,points=[[-1.,.5],[.3,.5],[.8,.5],[1.2,.5],[2.3,.5],[3.,.5]])]
        result=summarize([snapshot()],plans)
        for plan in result['plans']['rows']:
            self.assertAlmostEqual(plan['length_m'],4.)
            self.assertAlmostEqual(plan['length_m_by_status']['known'],1.)
            self.assertAlmostEqual(plan['length_m_by_status']['unknown'],1.)
            self.assertAlmostEqual(plan['length_m_by_status']['outside_grid'],2.)

    def test_rotated_quality_origin_is_respected(self):
        rotated=snapshot();rotated['origin']=[0.,0.,math.pi/2]
        result=summarize([rotated],[dict(t=.5,points=[[-.5,0.],[-.5,2.]])])
        self.assertAlmostEqual(result['plans']['rows'][0]['known_length_fraction'],.5)

    def test_missing_receipt_is_explicit_and_clock_reversal_is_not_reordered_away(self):
        row=snapshot();del row['received_ros']
        self.assertEqual(summarize([row])['availability_time'],'source_time_only_or_mixed')
        self.assertEqual(summarize([snapshot(.7),snapshot(.6)])['status'],'clock_discontinuity')


if __name__=='__main__':
    unittest.main(verbosity=2)
