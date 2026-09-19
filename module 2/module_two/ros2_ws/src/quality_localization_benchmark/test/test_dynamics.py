"""Fixed-event protocol tests without ROS processes or sampled robot truth."""
import copy
import math
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from quality_localization_benchmark import dynamics,worlds
from quality_localization_benchmark.sensor_bridge import EnvironmentModel,EventEpoch,FaultParameters
from test_worlds import shortest_grid_route


def metadata():
    layout=worlds.LAYOUTS['test_ring']
    return dict(map_id='test_ring',**copy.deepcopy(layout),rectangles=worlds.all_rectangles(layout))


class DynamicGeometryTest(unittest.TestCase):
    def test_partial_block_preserves_an_upper_route_and_full_block_disconnects(self):
        info=metadata()
        partial=dynamics.make_dynamic_config('partial_block',info)
        route=shortest_grid_route(info,info['rectangles']+partial['rectangles'])
        self.assertIsNotNone(route)
        crossings=[y for x,y in route if abs(x)<.2]
        self.assertTrue(crossings and all(y>2. for y in crossings))
        full=dynamics.make_dynamic_config('full_block',info)
        self.assertIsNone(shortest_grid_route(info,info['rectangles']+full['rectangles']))
        self.assertIsNotNone(shortest_grid_route(info,info['rectangles']))

    def test_sdf_has_real_collision_at_registered_bounds(self):
        for scenario in ('partial_block','full_block'):
            with self.subTest(scenario=scenario):
                config=dynamics.make_dynamic_config(scenario,metadata())
                root=ET.fromstring(dynamics.obstacle_sdf(config))
                self.assertEqual(root.findtext('model/static'),'true')
                self.assertEqual(root.find('model').get('name'),config['model_name'])
                recovered=[]
                for link in root.findall('model/link'):
                    x,y,z,*angles=[float(v) for v in link.findtext('pose').split()]
                    sx,sy,sz=[float(v) for v in link.findtext('collision/geometry/box/size').split()]
                    recovered.append([x-sx/2,x+sx/2,y-sy/2,y+sy/2])
                    self.assertAlmostEqual(z-sz/2,0.)
                    self.assertGreater(sz,1.)
                    self.assertTrue(all(angle==0 for angle in angles))
                for a,b in zip(recovered,config['rectangles']):
                    for x,y in zip(a,b):
                        self.assertAlmostEqual(x,y)

    def test_spawn_position_is_safe_from_frozen_start_reachability_bound(self):
        for scenario in ('partial_block','full_block'):
            config=dynamics.make_dynamic_config(scenario,metadata())
            self.assertGreater(config['spawn_safety']['clearance_margin_m'],2.6)
        changed=metadata();changed['start']=[-.2,-3.,0.]
        with self.assertRaises(ValueError):
            dynamics.make_dynamic_config('partial_block',changed)

    def test_nonregistered_map_and_invalid_seed_are_rejected(self):
        wrong=metadata();wrong['map_id']='dev_rooms'
        with self.assertRaises(ValueError):
            dynamics.make_dynamic_config('full_block',wrong)
        for seed in (-1,True,2**63):
            with self.subTest(seed=seed),self.assertRaises(ValueError):
                dynamics.make_dynamic_config('sensor_recovery',metadata(),seed)

    def test_configuration_changes_no_geometry_by_method_or_random_draw(self):
        a=dynamics.make_dynamic_config('full_block',metadata(),100)
        b=dynamics.make_dynamic_config('full_block',metadata(),102)
        self.assertEqual(a['rectangles'],b['rectangles'])
        self.assertEqual(a['activation_offset_sec'],8)
        self.assertEqual(a['release_offset_sec'],28)
        self.assertNotIn('method',a)
        self.assertEqual(a['sensor_environment']['zones'],[])


class DynamicScheduleTest(unittest.TestCase):
    def schedule(self,scenario='partial_block'):
        return dynamics.EventSchedule(dynamics.make_dynamic_config(scenario,metadata()))

    def test_start_is_once_and_cannot_refresh_epoch(self):
        schedule=self.schedule()
        self.assertIsNone(schedule.next_due(100.))
        self.assertEqual(schedule.start(100.),100.)
        self.assertEqual(schedule.start(109.),100.)
        self.assertIsNone(schedule.next_due(107.99))
        self.assertEqual(schedule.next_due(108.),0)

    def test_pending_request_does_not_claim_success_or_issue_duplicate(self):
        schedule=self.schedule();schedule.start(100.)
        schedule.requested(0,108.)
        self.assertIsNone(schedule.next_due(108.1))
        self.assertFalse(schedule.state(108.1)['activated'])
        self.assertTrue(schedule.state(108.1)['possibly_active_rectangles'])
        self.assertFalse(schedule.state(108.1)['active_rectangles'])
        with self.assertRaises(ValueError):
            schedule.requested(0,108.2)

    def test_unavailable_service_failure_does_not_fabricate_a_physical_request(self):
        schedule=self.schedule();schedule.start(100.)
        schedule.failed_without_request(0,108.2,'gazebo_service_unavailable')
        state=schedule.state(109.)
        self.assertIsNone(state['events'][0]['requested_ros'])
        self.assertFalse(state['possibly_active_rectangles'])
        self.assertFalse(state['valid_evidence'])

    def test_activation_cannot_be_requested_before_the_fixed_offset(self):
        schedule=self.schedule();schedule.start(100.)
        with self.assertRaises(ValueError):
            schedule.requested(0,107.99)

    def test_actual_request_response_times_survive_in_full_state(self):
        schedule=self.schedule();schedule.start(100.)
        schedule.requested(0,108.01);schedule.finished(0,108.04,True,'spawned')
        self.assertIsNone(schedule.next_due(127.99))
        self.assertEqual(schedule.next_due(128.),1)
        schedule.requested(1,128.02);schedule.finished(1,128.07,True,'deleted')
        state=schedule.state(129.)
        self.assertTrue(state['completed'] and state['valid_evidence'])
        self.assertEqual(state['events'][0]['requested_ros'],108.01)
        self.assertEqual(state['events'][0]['succeeded_ros'],108.04)
        self.assertEqual(state['events'][1]['succeeded_ros'],128.07)
        self.assertFalse(state['active_rectangles'])
        self.assertFalse(state['possibly_active_rectangles'])
        self.assertGreaterEqual(len(state['history']),5)
        state['events'][0]['success']=False
        self.assertTrue(schedule.state(129.)['events'][0]['success'])

    def test_spawn_timeout_is_invalid_but_release_still_cleans_own_possible_model(self):
        schedule=self.schedule();schedule.start(100.)
        schedule.requested(0,108.);schedule.finished(0,108.2,False,'timeout_effect_unknown')
        self.assertFalse(schedule.state(109.)['valid_evidence'])
        self.assertEqual(schedule.next_due(128.),1)
        schedule.requested(1,128.);schedule.finished(1,128.1,True,'deleted')
        self.assertFalse(schedule.state(129.)['completed'])
        self.assertFalse(schedule.state(129.)['possibly_active_rectangles'])

    def test_missed_activation_is_not_spawned_late_under_the_robot(self):
        schedule=self.schedule();schedule.start(100.)
        self.assertIsNone(schedule.next_due(110.))
        self.assertEqual(schedule.events[0]['state'],'failed')
        self.assertIn('activation_deadline_missed',schedule.invalid_reasons)
        self.assertIsNone(schedule.next_due(128.))

    def test_clock_rewind_invalidates_sequence_without_restarting(self):
        schedule=self.schedule();schedule.start(100.)
        schedule.observe_clock(107.);schedule.observe_clock(1.)
        self.assertIsNone(schedule.next_due(108.))
        self.assertIn('clock_rewind',schedule.state(108.)['invalid_reasons'])
        self.assertEqual(schedule.start(1.),100.)


class DynamicSensorEpochTest(unittest.TestCase):
    def model(self):
        return EnvironmentModel(dynamics.make_dynamic_config('sensor_recovery',metadata(),100)['sensor_environment'])

    def test_sensor_events_wait_for_trigger_and_use_relative_source_time(self):
        model=self.model();gate=EventEpoch(require_start=True)
        before=model.parameters_at(0,0,1000.,event_epoch=gate.epoch or 0.,events_enabled=gate.enabled)
        self.assertEqual(before,FaultParameters())
        self.assertEqual(gate.start(100.),100.)
        self.assertEqual(gate.start(120.),100.)
        self.assertEqual(model.parameters_at(0,0,107.99,event_epoch=gate.epoch),FaultParameters())
        active=model.parameters_at(0,0,108.,event_epoch=gate.epoch)
        self.assertAlmostEqual(active.laser_sigma,.08)
        self.assertAlmostEqual(active.dropout,.45)
        self.assertEqual(active.visual_blur,9)
        self.assertAlmostEqual(active.exposure,.35)
        self.assertEqual(model.parameters_at(0,0,128.,event_epoch=gate.epoch),FaultParameters())

    def test_default_epoch_zero_preserves_existing_static_absolute_event_semantics(self):
        gate=EventEpoch()
        self.assertTrue(gate.enabled)
        self.assertEqual(gate.epoch,0.)
        model=self.model()
        self.assertEqual(model.parameters_at(0,0,7.99),FaultParameters())
        self.assertGreater(model.parameters_at(0,0,8.).dropout,0)
        self.assertEqual(model.parameters_at(0,0,28.),FaultParameters())

    def test_epoch_rewind_disables_events_instead_of_silent_replay(self):
        gate=EventEpoch(True);gate.start(100.);gate.clock_rewound()
        self.assertFalse(gate.enabled)
        self.assertTrue(gate.clock_invalid)
        with self.assertRaises(ValueError):
            gate.start(1.)
        self.assertEqual(self.model().parameters_at(0,0,108.,event_epoch=gate.epoch,
                                                   events_enabled=gate.enabled),FaultParameters())

    def test_recovery_configuration_has_no_quality_scores_or_error_labels(self):
        config=dynamics.make_dynamic_config('sensor_recovery',metadata())
        self.assertEqual(config['rectangles'],[])
        self.assertTrue(config['require_event_start'])
        self.assertEqual(config['sensor_environment']['zones'][0]['bounds'],[-8,8,-6,6])
        self.assertEqual(set(config['sensor_environment']['events'][0]),
                         {'zone','start','end','laser_sigma','dropout','visual_blur','exposure'})


if __name__=='__main__':
    unittest.main(verbosity=2)
