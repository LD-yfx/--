"""Independent geometry checks; these do not establish live navigation success.

Raster occupancy uses cell centres. Continuous SDF geometry is checked separately
with a circular footprint and a conservative cell-radius margin for connectivity.
No ROS node, Gazebo process, or generated localization-quality field is involved.
"""
from collections import deque
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE / 'quality_localization_benchmark/worlds.py'
SPEC = importlib.util.spec_from_file_location('benchmark_worlds', SOURCE)
worlds = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worlds)
WORKSPACE = PACKAGE.parents[3]
REFERENCE = (WORKSPACE / 'module_one_reference/localization_quality_final/'
             'simulation/robot/sensor_robot.urdf.xacro')


def sdf_rectangles(document):
    """Recover continuous collision bounds from XML, independent of layout data."""
    root = ET.fromstring(document)
    rectangles = []
    for model in root.findall('./world/model'):
        if model.get('name') == 'ground':
            continue
        pose = np.array([float(v) for v in model.findtext('pose').split()])
        for collision in model.findall('./link/collision'):
            if collision.find('pose') is not None:
                raise AssertionError('Rotated/offset collision needs a new test oracle')
            size = [float(v) for v in collision.findtext('./geometry/box/size').split()]
            if not np.allclose(pose[3:], 0):
                raise AssertionError('Rotated boxes need a new test oracle')
            rectangles.append([pose[0] - size[0] / 2, pose[0] + size[0] / 2,
                               pose[1] - size[1] / 2, pose[1] + size[1] / 2])
    return rectangles


def clearance(x, y, rectangles):
    """Exact point-to-axis-aligned-box distance in the world XY plane."""
    result = np.full(np.broadcast(x, y).shape, np.inf)
    for left, right, bottom, top in rectangles:
        dx = np.maximum(np.maximum(left - x, x - right), 0)
        dy = np.maximum(np.maximum(bottom - y, y - top), 0)
        result = np.minimum(result, np.hypot(dx, dy))
    return result


def shortest_grid_route(layout, rectangles, forbidden_zone=None):
    """Four-connected route with disk clearance for every selected cell area."""
    resolution = .1
    x0, x1, y0, y1 = layout['bounds']
    width, height = round((x1 - x0) / resolution), round((y1 - y0) / resolution)
    xx, yy = np.meshgrid(x0 + (np.arange(width) + .5) * resolution,
                         y0 + (np.arange(height) + .5) * resolution)
    safe = clearance(xx, yy, rectangles) > .32 + resolution / math.sqrt(2)
    if forbidden_zone is not None:
        left, right, bottom, top = forbidden_zone
        safe &= ~((xx >= left) & (xx <= right) & (yy >= bottom) & (yy <= top))

    def cell(point):
        return (int((point[1] - y0) / resolution), int((point[0] - x0) / resolution))

    start, goal = cell(layout['start']), cell(layout['goal'])
    if not safe[start] or not safe[goal]:
        return None
    queue = deque([start])
    parent = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            route = []
            while current is not None:
                route.append(current)
                current = parent[current]
            return [(x0 + (x + .5) * resolution, y0 + (y + .5) * resolution)
                    for y, x in route[::-1]]
        y, x = current
        for nxt in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            ny, nx = nxt
            if (0 <= ny < height and 0 <= nx < width and safe[nxt]
                    and nxt not in parent):
                parent[nxt] = current
                queue.append(nxt)
    return None


class WorldGeometryTest(unittest.TestCase):
    def test_sdf_collision_bounds_match_layout_and_cell_centres(self):
        for map_id, layout in worlds.LAYOUTS.items():
            with self.subTest(map=map_id):
                boxes = sdf_rectangles(worlds.gazebo_world(map_id, layout))
                np.testing.assert_allclose(boxes, worlds.all_rectangles(layout), atol=1e-12)
                x0, x1, y0, y1 = layout['bounds']
                xx, yy = np.meshgrid(x0 + (np.arange(round((x1 - x0) / .1)) + .5) * .1,
                                     y0 + (np.arange(round((y1 - y0) / .1)) + .5) * .1)
                # Some centres lie on a box boundary. Floating serialization
                # may move that boundary <1e-14 m; compare away from it and
                # require any difference to be solely that numerical surface.
                blocked = np.zeros_like(xx, dtype=bool)
                near_boundary = np.zeros_like(blocked)
                for left, right, bottom, top in boxes:
                    blocked |= (xx >= left) & (xx <= right) & (yy >= bottom) & (yy <= top)
                    near_boundary |= (((abs(xx - left) < 1e-12) | (abs(xx - right) < 1e-12))
                                      & (yy >= bottom - 1e-12) & (yy <= top + 1e-12))
                    near_boundary |= (((abs(yy - bottom) < 1e-12) | (abs(yy - top) < 1e-12))
                                      & (xx >= left - 1e-12) & (xx <= right + 1e-12))
                actual = worlds.occupancy(layout)
                self.assertFalse(np.any((actual != blocked) & ~near_boundary))

    def test_visual_panels_have_no_collision_and_real_sensor_geometry_exists(self):
        for map_id, layout in worlds.LAYOUTS.items():
            with self.subTest(map=map_id):
                root = ET.fromstring(worlds.gazebo_world(map_id, layout))
                panels = [model for model in root.findall('./world/model')
                          if model.get('name').startswith('panel_')]
                self.assertEqual(len(panels), 48)
                self.assertTrue(all(model.findall('./link/visual') for model in panels))
                self.assertFalse(any(model.findall('./link/collision') for model in panels))
                self.assertEqual(len(sdf_rectangles(worlds.gazebo_world(map_id, layout))),
                                 4 + len(layout['interior']))

    def test_roundtrip_is_connected_with_continuous_robot_clearance(self):
        for map_id, layout in worlds.LAYOUTS.items():
            with self.subTest(map=map_id):
                boxes = sdf_rectangles(worlds.gazebo_world(map_id, layout))
                for endpoint in (layout['start'], layout['goal']):
                    self.assertGreater(float(clearance(endpoint[0], endpoint[1], boxes)), .32)
                route = shortest_grid_route(layout, boxes)
                self.assertIsNotNone(route, 'No footprint-safe start-to-goal route')
                # Reversing this geometric path proves roundtrip connectivity,
                # not AMCL/Nav2 convergence or a dynamically feasible turn.
                for point in route:
                    self.assertGreater(float(clearance(point[0], point[1], boxes)), .32)

    def test_each_layout_has_a_route_avoiding_measurement_degradation(self):
        for map_id, layout in worlds.LAYOUTS.items():
            with self.subTest(map=map_id):
                boxes = sdf_rectangles(worlds.gazebo_world(map_id, layout))
                self.assertIsNotNone(shortest_grid_route(layout, boxes, layout['degradation']),
                                     'A quality penalty has no route-choice alternative')

    def test_development_calibration_visits_degradation_and_revisits_safely(self):
        for map_id, layout in worlds.LAYOUTS.items():
            with self.subTest(map=map_id):
                if layout['split'] == 'test':
                    self.assertNotIn('calibration_waypoints', layout)
                    continue
                waypoints = layout['calibration_waypoints']
                self.assertEqual(len(waypoints), 4)
                self.assertEqual(waypoints[-1], layout['start'][:2])
                self.assertEqual(waypoints[1], layout['goal'])
                self.assertEqual(waypoints[0], waypoints[2])
                left, right, bottom, top = layout['degradation']
                self.assertTrue(left <= waypoints[0][0] < right and
                                bottom <= waypoints[0][1] < top)
                boxes = sdf_rectangles(worlds.gazebo_world(map_id, layout))
                previous = layout['start']
                for goal in waypoints:
                    self.assertGreater(float(clearance(goal[0], goal[1], boxes)), .32)
                    self.assertIsNotNone(shortest_grid_route(dict(layout, start=previous, goal=goal), boxes))
                    previous = goal

    def test_generated_map_orientation_metadata_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            worlds.write_worlds(target)
            manifest = json.loads((target / 'manifest.json').read_text())
            self.assertEqual(set(manifest['layouts']), set(worlds.LAYOUTS))
            expected_files = {str(path.relative_to(target)) for path in target.rglob('*')
                              if path.is_file() and path.name != 'manifest.json'}
            self.assertEqual(set(manifest['sha256']), expected_files)
            for name, digest in manifest['sha256'].items():
                self.assertEqual(hashlib.sha256((target / name).read_bytes()).hexdigest(), digest)
            for map_id, layout in worlds.LAYOUTS.items():
                with self.subTest(map=map_id):
                    folder = target / map_id
                    magic, size, maximum, payload = (folder / 'map.pgm').read_bytes().split(b'\n', 3)
                    width, height = [int(value) for value in size.split()]
                    self.assertEqual((magic, maximum), (b'P5', b'255'))
                    image = np.frombuffer(payload, np.uint8).reshape(height, width)
                    np.testing.assert_array_equal(image[::-1] == 0, worlds.occupancy(layout))
                    metadata = json.loads((folder / 'metadata.json').read_text())
                    self.assertEqual(metadata['world_to_map'], [0, 0, 0])
                    self.assertEqual(metadata['mission_waypoints'], [layout['goal'], layout['start'][:2]])
                    self.assertFalse(metadata['quality_values_preassigned'])
                    self.assertEqual(metadata['quality_width'] * metadata['quality_resolution'], width * .1)
                    self.assertEqual(metadata['quality_height'] * metadata['quality_resolution'], height * .1)
                    boxes=sdf_rectangles((folder/'world.sdf').read_text())
                    for route_key,length_key,limit_key in (
                            ('mission_waypoints','reference_mission_length_m','mission_time_limit_sec'),
                            ('calibration_waypoints','reference_calibration_length_m','calibration_time_limit_sec')):
                        if route_key not in metadata:
                            continue
                        previous=layout['start']
                        independent_length=0.
                        for goal in metadata[route_key]:
                            route=shortest_grid_route(dict(layout,start=previous,goal=goal),boxes)
                            self.assertIsNotNone(route)
                            independent_length+=(len(route)-1)*.1
                            independent_length+=math.dist(previous[:2],route[0])+math.dist(goal,route[-1])
                            previous=goal
                        self.assertAlmostEqual(metadata[length_key],independent_length,places=9)
                        self.assertAlmostEqual(metadata[limit_key],max(90,4*independent_length/.4+30),places=9)
                    self.assertFalse(metadata['reference_path_parameters']['continuous_shortest_path'])
                    yaml = (folder / 'map.yaml').read_text()
                    self.assertIn('resolution: 0.1\n', yaml)
                    self.assertIn(f"origin: [{layout['bounds'][0]}, {layout['bounds'][2]}, 0.0]", yaml)

    def test_conditions_change_measurements_without_quality_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            worlds.write_worlds(tmp)
            for map_id in worlds.LAYOUTS:
                for condition in worlds.CONDITIONS:
                    with self.subTest(map=map_id, condition=condition):
                        config = json.loads((Path(tmp) / map_id / (condition + '.json')).read_text())
                        self.assertEqual(set(config), {'version', 'seed', 'zones', 'events'})
                        if condition == 'normal':
                            self.assertEqual(config['zones'], [])
                            continue
                        zone, = config['zones']
                        self.assertEqual(set(zone), {'name', 'bounds', 'laser_sigma', 'dropout',
                                                    'visual_blur', 'exposure'})
                        self.assertEqual(zone['laser_sigma'] > 0, condition in ('laser', 'combined'))
                        self.assertEqual(zone['dropout'] > 0, condition in ('laser', 'combined'))
                        self.assertEqual(zone['visual_blur'] > 0, condition in ('visual', 'combined'))
                        self.assertEqual(zone['exposure'] < 1, condition in ('visual', 'combined'))


class RobotIsolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.tmp.name) / 'robot.xacro'
        worlds.make_robot(REFERENCE, cls.output)
        cls.root = ET.parse(cls.output).getroot()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_wheel_encoder_not_world_pose_owns_navigation_odometry(self):
        drive = self.root.find("gazebo/plugin[@name='diff_drive']")
        self.assertEqual([v.text for v in drive.findall('odometry_source')], ['0'])
        self.assertEqual(drive.findtext('publish_odom_tf'), 'true')
        self.assertEqual(drive.findtext('odometry_frame'), 'odom')
        self.assertEqual(drive.findtext('robot_base_frame'), 'base_footprint')
        self.assertNotIn('ground_truth', ET.tostring(drive, encoding='unicode'))
        truth = self.root.find("gazebo/plugin[@name='ground_truth']")
        self.assertEqual(truth.findtext('frame_name'), 'world')
        self.assertEqual(float(truth.findtext('gaussian_noise')), 0)
        self.assertEqual(truth.findtext('topic_name'), 'ground_truth/odom')
        self.assertEqual(truth.findtext('body_name'), 'base_footprint')
        self.assertFalse(truth.findall('publish_odom_tf'))
        self.assertEqual({plugin.get('filename') for plugin in self.root.findall('.//plugin')},
                         {'libgazebo_ros_diff_drive.so', 'libgazebo_ros_p3d.so',
                          'libgazebo_ros_ray_sensor.so', 'libgazebo_ros_camera.so'})

    def test_real_laser_and_camera_are_raw_fault_bridge_inputs(self):
        scan = self.root.find(".//sensor[@type='ray']")
        self.assertEqual(scan.findtext('update_rate'), '15')
        self.assertEqual(scan.findtext('ray/scan/horizontal/samples'), '720')
        self.assertEqual(scan.findtext('plugin/ros/remapping'), '~/out:=sim/scan')
        camera = self.root.find(".//sensor[@type='camera']")
        self.assertEqual(camera.findtext('update_rate'), '10')
        self.assertEqual(camera.findtext('camera/image/width'), '320')
        self.assertEqual(camera.findtext('camera/image/height'), '240')
        self.assertEqual(camera.findtext('plugin/camera_name'), 'front_camera')
        self.assertFalse(self.root.findall(".//sensor[@type='depth']"))

    def test_wheel_and_caster_contact_heights_and_collision_radius(self):
        macro = self.root.find('{http://www.ros.org/wiki/xacro}macro')
        self.assertEqual(macro.find('joint/axis').get('xyz'), '0 1 0')
        self.assertEqual(macro.find('joint/origin').get('rpy'), '0 0 0')
        wheel_rotation = [float(v) for v in macro.find('link/collision/origin').get('rpy').split()]
        self.assertAlmostEqual(abs(wheel_rotation[0]), math.pi / 2, places=9)
        self.assertAlmostEqual(wheel_rotation[1], 0)
        self.assertAlmostEqual(wheel_rotation[2], 0)
        base_z = float(self.root.find("joint[@name='base_footprint_joint']/origin").get('xyz').split()[2])
        wheel_radius = float(self.root.find("{http://www.ros.org/wiki/xacro}property[@name='wheel_radius']").get('value'))
        self.assertAlmostEqual(base_z - wheel_radius, 0)
        for name in ('rear_caster', 'front_caster'):
            caster_z = float(self.root.find(f"joint[@name='{name}_joint']/origin").get('xyz').split()[2])
            radius = float(self.root.find(f"link[@name='{name}']/collision/geometry/sphere").get('radius'))
            self.assertAlmostEqual(base_z + caster_z - radius, 0)
        body_size = [float(v) for v in self.root.find("link[@name='base_link']/collision/geometry/box").get('size').split()]
        self.assertLess(math.hypot(body_size[0] / 2, body_size[1] / 2), .31)
        self.assertGreater(base_z + .05 - body_size[2] / 2, 0)

    def test_checked_in_robot_matches_generator(self):
        self.assertEqual((PACKAGE / 'robot/research_robot.urdf.xacro').read_bytes(), self.output.read_bytes())


if __name__ == '__main__':
    unittest.main(verbosity=2)
