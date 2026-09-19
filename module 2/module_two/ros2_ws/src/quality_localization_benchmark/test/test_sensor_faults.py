"""Pure algorithm tests: no ROS graph, Gazebo, truth-error or quality-score stubs."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'quality_localization_benchmark/sensor_bridge.py'
SPEC = importlib.util.spec_from_file_location('benchmark_sensor_faults', SOURCE)
faults = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = faults
SPEC.loader.exec_module(faults)


class SensorFaultTest(unittest.TestCase):
    def model(self):
        return faults.EnvironmentModel({
            'version': 1, 'seed': 42,
            'zones': [{'name': 'area', 'bounds': [-2, 2, -1, 1],
                       'laser_sigma': 0.03, 'dropout': 0.1, 'visual_blur': 5, 'exposure': 0.6}],
            'events': [{'zone': 'area', 'start': 20, 'end': 40, 'dropout': 0.3}]})

    def test_empty_configuration_is_transparent_without_truth(self):
        model = faults.EnvironmentModel()
        self.assertFalse(model.needs_truth)
        self.assertEqual(model.parameters_at(0, 0, 1), faults.FaultParameters())

    def test_spatial_boundaries_are_half_open(self):
        model = self.model()
        self.assertAlmostEqual(model.parameters_at(-2, -1, 5).dropout, 0.1)
        self.assertEqual(model.parameters_at(2, 0, 5), faults.FaultParameters())
        self.assertEqual(model.parameters_at(0, 1, 5), faults.FaultParameters())

    def test_events_override_only_listed_fields_and_expire(self):
        model = self.model()
        self.assertAlmostEqual(model.parameters_at(0, 0, 19.99).dropout, 0.1)
        active = model.parameters_at(0, 0, 20)
        self.assertAlmostEqual(active.dropout, 0.3)
        self.assertAlmostEqual(active.laser_sigma, 0.03)
        self.assertEqual(active.visual_blur, 5)
        self.assertAlmostEqual(model.parameters_at(0, 0, 40).dropout, 0.1)

    def test_invalid_and_ambiguous_scenarios_are_rejected(self):
        for config in (
            {'seed': -1}, {'version': 2}, {'truth_error_scale': 1},
            {'zones': [{'name': 'bad', 'bounds': [0, 0, 0, 1]}]},
            {'zones': [{'name': 'a', 'bounds': [0, 2, 0, 2]},
                       {'name': 'b', 'bounds': [1, 3, 1, 3]}]},
            {'events': [{'zone': 'missing', 'start': 1, 'end': 2}]},
            {'zones': [{'name': 'a', 'bounds': [0, 2, 0, 2]}],
             'events': [{'zone': 'a', 'start': 1, 'end': 3},
                        {'zone': 'a', 'start': 2, 'end': 4}]},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                faults.EnvironmentModel(config)

    def test_parameter_domains_and_explicit_seed_override(self):
        for values in ({'laser_sigma': -1}, {'dropout': 1.1}, {'visual_blur': 4},
                       {'visual_blur': 3.0}, {'exposure': -1}, {'exposure': float('nan')}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                faults.FaultParameters(**values)
        self.assertEqual(faults.EnvironmentModel({'seed': 42}, seed_override=13).seed, 13)

    def test_laser_passthrough_preserves_invalid_values(self):
        original = np.array([1.0, float('inf'), float('nan'), 0.01, 25.0])
        result = faults.apply_laser_fault(original, 0.1, 20.0, faults.FaultParameters(), 42, 1)
        np.testing.assert_equal(result, original)

    def test_noise_reproducibility_does_not_depend_on_call_order(self):
        data = np.full(500, 5.0)
        fault = faults.FaultParameters(laser_sigma=0.1, dropout=0.2)
        first = faults.apply_laser_fault(data, 0.1, 20.0, fault, 42, 10_000_000_000)
        faults.apply_laser_fault(data, 0.1, 20.0, fault, 42, 900_000_000_000)
        repeat = faults.apply_laser_fault(data, 0.1, 20.0, fault, 42, 10_000_000_000)
        np.testing.assert_array_equal(first, repeat)
        changed = faults.apply_laser_fault(data, 0.1, 20.0, fault, 43, 10_000_000_000)
        self.assertFalse(np.array_equal(first, changed))

    def test_noise_and_dropout_have_independent_component_streams(self):
        data = np.full(500, 5.0)
        noise = faults.apply_laser_fault(data, 0.1, 20.0,
                                        faults.FaultParameters(laser_sigma=0.1), 42, 123)
        combined = faults.apply_laser_fault(data, 0.1, 20.0,
                                           faults.FaultParameters(laser_sigma=0.1, dropout=0.4), 42, 123)
        retained = np.isfinite(combined)
        np.testing.assert_array_equal(combined[retained], noise[retained])

    def test_noise_distribution_and_dropout_rate(self):
        data = np.full(20000, 5.0)
        noise = faults.apply_laser_fault(data, 0.1, 20.0,
                                        faults.FaultParameters(laser_sigma=0.05), 9, 555) - data
        self.assertLess(abs(float(noise.mean())), 0.002)
        self.assertLess(abs(float(noise.std()) - 0.05), 0.002)
        dropped = faults.apply_laser_fault(data, 0.1, 20.0,
                                          faults.FaultParameters(dropout=0.25), 9, 555)
        self.assertLess(abs(float(np.isinf(dropped).mean()) - 0.25), 0.02)

    def test_laser_noise_is_bounded_and_dropouts_are_no_return(self):
        data = [0.1, 20, float('inf'), float('nan')]
        result = faults.apply_laser_fault(data, 0.1, 20,
                                         faults.FaultParameters(laser_sigma=100), 5, 99)
        self.assertTrue(np.all((result[:2] >= 0.1) & (result[:2] <= 20)))
        self.assertTrue(np.isinf(result[2]) and np.isnan(result[3]))
        result = faults.apply_laser_fault([1, 2], 0.1, 20,
                                         faults.FaultParameters(dropout=1), 5, 99)
        self.assertTrue(np.isinf(result).all())

    def test_exposure_preserves_shape_dtype_and_does_not_modify_input(self):
        image = np.array([[[0, 100, 200], [30, 40, 255]]], dtype=np.uint8)
        saved = image.copy()
        result = faults.apply_image_fault(image, faults.FaultParameters(exposure=2))
        np.testing.assert_array_equal(result, [[[0, 200, 255], [60, 80, 255]]])
        np.testing.assert_array_equal(image, saved)
        self.assertEqual(result.dtype, image.dtype)
        dark = faults.apply_image_fault(image, faults.FaultParameters(exposure=0))
        self.assertEqual(int(dark.sum()), 0)

    def test_blur_reduces_actual_image_edge_energy(self):
        import cv2
        image = np.zeros((64, 64), dtype=np.uint8)
        image[:, 32:] = 255
        blurred = faults.apply_image_fault(image, faults.FaultParameters(visual_blur=9))
        self.assertLess(cv2.Laplacian(blurred, cv2.CV_64F).var(),
                        cv2.Laplacian(image, cv2.CV_64F).var())

    def test_truth_history_never_uses_future_or_stale_pose(self):
        history = faults.TruthHistory(max_age_sec=0.2)
        history.push(1_000_000_000, 1, 2)
        history.push(1_100_000_000, 3, 4)
        self.assertEqual(history.lookup(1_050_000_000), (1, 2))
        self.assertIsNone(history.lookup(900_000_000))
        self.assertIsNone(history.lookup(1_400_000_000))
        self.assertTrue(history.covers(1_050_000_000))
        self.assertFalse(history.covers(1_200_000_000))

    def test_truth_clock_rewind_clears_previous_epoch(self):
        history = faults.TruthHistory(capacity=2)
        history.push(10_000_000_000, 1, 2)
        self.assertTrue(history.push(1_000_000_000, 7, 8))
        self.assertEqual(history.lookup(1_000_000_000), (7, 8))
        self.assertIsNone(history.lookup(10_000_000_000))
        self.assertEqual(len(history.entries), 1)


if __name__ == '__main__':
    unittest.main()
