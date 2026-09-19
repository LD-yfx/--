"""Independent source-time contact, stop and recovery evidence regressions."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from quality_localization_benchmark.dynamic_evaluation import summarize_dynamic_episode,swept_contacts


def events(sensor=False):
    kinds=('sensor_fault_begin','sensor_fault_end') if sensor else ('spawn_obstacle','delete_obstacle')
    return dict(scenario='sensor_recovery' if sensor else 'full_block',epoch=0.,
        valid_evidence=True,invalid_reasons=[],scheduled_activation_ros=8.,scheduled_release_ros=28.,
        rectangles=[] if sensor else [[-.15,.15,-5.8,5.8]],
        events=[dict(kind=kinds[0],state='succeeded',success=True,requested_ros=8.,succeeded_ros=8.1),
                dict(kind=kinds[1],state='succeeded',success=True,requested_ros=28.,succeeded_ros=28.1)])


def poses(position=None,end=35.):
    position=position or (lambda t:(-2.,0.,0.))
    return [dict(t=i*.02,x=position(i*.02)[0],y=position(i*.02)[1],yaw=position(i*.02)[2],frame='world')
            for i in range(round(end/.02)+1)]


def evaluate(state=None,truth=None,commands=(),start=0.,end=35.,static=()):
    return summarize_dynamic_episode(events() if state is None else state,
        poses() if truth is None else truth,commands,start,end,
        dict(robot_collision_radius=.1,rectangles=list(static)))


class DynamicEvaluationTest(unittest.TestCase):
    def test_service_request_response_brackets_are_kept_distinct(self):
        result=evaluate()
        timing=result['obstacle_timing']
        self.assertEqual(timing['possible_presence_interval'],[8.,28.1])
        self.assertEqual(timing['confirmed_presence_interval'],[8.1,28.])
        self.assertEqual(timing['uncertain_intervals'],[[8.,8.1],[28.,28.1]])
        self.assertTrue(result['event_timing_valid'])

    def test_contact_uses_possible_presence_not_only_success_time(self):
        result=evaluate(truth=poses(lambda t:(0.,0.,0.)))
        contacts=result['contacts']
        self.assertEqual(contacts['dynamic_contact_count'],1)
        self.assertAlmostEqual(contacts['dynamic_contact_intervals'][0][0],8.)
        self.assertAlmostEqual(contacts['dynamic_contact_intervals'][0][1],28.1)
        self.assertEqual(len(contacts['dynamic_contacts_during_uncertain_physics']),2)

    def test_static_and_dynamic_contacts_are_unioned_without_double_counting(self):
        result=evaluate(truth=poses(lambda t:(0.,0.,0.)),static=[[-.15,.15,-5.8,5.8]])
        self.assertEqual(result['contacts']['static_contact_count'],1)
        self.assertEqual(result['contacts']['dynamic_contact_count'],1)
        self.assertEqual(result['contacts']['combined_contact_count'],1)
        self.assertEqual(result['contacts']['combined_contact_intervals'],[[0.,35.]])

    def test_failed_delete_keeps_possible_obstacle_until_mission_end(self):
        state=events();state['events'][1].update(success=False,state='failed',succeeded_ros=None)
        result=evaluate(state,poses(lambda t:(0.,0.,0.)))
        self.assertEqual(result['obstacle_timing']['possible_presence_interval'],[8.,35.])
        self.assertFalse(result['event_timing_valid'])
        self.assertIsNone(result['recovery']['release_confirmed_ros'])

    def test_physical_stop_and_recovery_are_distinct_from_commanded_motion(self):
        truth=poses(lambda t:(-2.+.1*min(t,9.)+.1*max(0.,t-29.),0.,0.))
        commands=[dict(t=i*.1,vx=.2,wz=0.) for i in range(351)]
        result=evaluate(truth=truth,commands=commands)
        closure=result['possible_closure_motion']
        self.assertAlmostEqual(closure['stop_onset_delay_sec'],1.,places=7)
        self.assertAlmostEqual(closure['stop_confirmed_delay_sec'],2.,places=7)
        self.assertAlmostEqual(result['late_closure_motion']['maximum_linear_speed_mps'],0.)
        self.assertGreater(result['late_closure_commands']['nonzero_messages'],0)
        self.assertAlmostEqual(result['recovery']['motion_onset_delay_sec'],.9,places=7)
        self.assertTrue(result['recovery']['stopped_for_one_second_before_release'])

    def test_rotation_in_place_is_not_a_stop(self):
        result=evaluate(truth=poses(lambda t:(-2.,0.,.1*t)))
        self.assertIsNone(result['possible_closure_motion']['first_sustained_stop'])
        self.assertGreater(result['late_closure_motion']['maximum_angular_speed_radps'],.09)

    def test_already_stopped_before_activation_is_not_claimed_as_reaction(self):
        result=evaluate()
        self.assertTrue(result['already_stopped_before_activation'])
        self.assertAlmostEqual(result['possible_closure_motion']['stop_onset_delay_sec'],0.)
        self.assertIsNone(result['recovery']['motion_onset_delay_sec'])

    def test_sparse_truth_is_not_interpolated_across_a_long_outage(self):
        truth=[dict(t=8.,x=-1.,y=0.,yaw=0.),dict(t=9.,x=1.,y=0.,yaw=0.)]
        result=evaluate(truth=truth)
        self.assertEqual(result['mission_kinematic_coverage'],0.)
        self.assertIsNone(result['possible_closure_motion']['first_sustained_stop'])
        self.assertEqual(result['contacts']['dynamic_contact_count'],0)

    def test_isolated_contact_sample_is_preserved_despite_unavailable_motion(self):
        result=evaluate(truth=[dict(t=10.,x=0.,y=0.,yaw=0.)])
        self.assertEqual(result['contacts']['dynamic_contact_intervals'],[[10.,10.]])
        self.assertEqual(result['contacts']['dynamic_contact_count'],1)
        self.assertEqual(result['mission_kinematic_coverage'],0.)

    def test_no_truth_or_commands_are_unknown_not_zero_risk(self):
        result=evaluate(truth=[])
        self.assertEqual(result['status'],'truth_unavailable')
        self.assertIsNone(result['contacts']['combined_contact_count'])
        self.assertIsNone(result['closure_commands']['nonzero_messages'])

    def test_truth_clock_rewind_is_not_sorted_away(self):
        truth=[dict(t=t,x=-2.,y=0.,yaw=0.) for t in (8.,9.,8.5)]
        result=evaluate(truth=truth)
        self.assertEqual(result['truth_statistics']['time_reversals'],1)
        self.assertFalse(result['contacts']['available'])

    def test_future_events_not_yet_due_do_not_invalidate_early_algorithm_failure(self):
        state=events()
        for event in state['events']:
            event.update(state='scheduled',success=None,requested_ros=None,succeeded_ros=None)
        result=evaluate(state,poses(end=5.),end=5.)
        self.assertTrue(result['event_timing_valid'])
        self.assertFalse(result['sequence_completed'])
        self.assertIsNone(result['obstacle_timing']['possible_presence_interval'])

    def test_reply_before_request_and_early_activation_are_rejected(self):
        state=events();state['events'][0]['succeeded_ros']=7.9
        result=evaluate(state)
        self.assertFalse(result['event_timing_valid'])
        self.assertIn('spawn_reply_before_request',result['event_timing_errors'])
        self.assertIn('activation_response_before_registered_offset',result['event_timing_errors'])

    def test_sensor_schedule_success_never_proves_processed_sensor_faults(self):
        result=evaluate(events(sensor=True))
        self.assertTrue(result['event_timing_valid'])
        self.assertTrue(result['sequence_completed'])
        self.assertFalse(result['sensor_effect_evidence']['actual_sensor_effect_verified'])
        self.assertEqual(result['sensor_effect_evidence']['status'],
                         'requires_independent_bridge_counters_and_source_frames')
        self.assertIsNone(result['obstacle_timing']['possible_presence_interval'])

    def test_swept_circle_catches_crossing_and_respects_rounded_corners(self):
        rectangles=[[0.,1.,0.,1.]]
        self.assertTrue(swept_contacts((-1.,.5),(2.,.5),rectangles,.1))
        self.assertFalse(swept_contacts((-.09,-.09),(-.09,-.09),rectangles,.1))
        self.assertTrue(swept_contacts((-.07,-.07),(-.07,-.07),rectangles,.1))


if __name__=='__main__':
    unittest.main(verbosity=2)
