#!/usr/bin/env python3
"""Generate the 27 deterministic Gazebo Classic worlds and run plans."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys


SCENES = ("corridor", "hall", "outdoor")
LIGHTING = {
    "dark": (0.04, 0.04, 0.04, 1.0),
    "normal": (0.65, 0.65, 0.65, 1.0),
    "strong": (1.0, 1.0, 1.0, 1.0),
}
RUN_VARIANTS = (
    ("low", "normal", "route_a", 0.40),
    ("medium", "visual_degraded", "route_b", 0.70),
    ("high", "laser_degraded", "route_c", 1.00),
)


def box(name, x, y, z, size_x, size_y, size_z, color="0.65 0.65 0.65 1"):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose>
      <link name="link">
        <collision name="collision"><geometry><box><size>{size_x} {size_y} {size_z}</size></box></geometry></collision>
        <visual name="visual">
          <geometry><box><size>{size_x} {size_y} {size_z}</size></box></geometry>
          <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>
        </visual>
      </link>
    </model>"""


def scene_models(scene, complexity, rng):
    models = []
    if scene == "corridor":
        models.extend((
            box("wall_left", 0, 2.0, 1.5, 40, 0.2, 3),
            box("wall_right", 0, -2.0, 1.5, 40, 0.2, 3),
            box("wall_end", 20, 0, 1.5, 0.2, 4, 3),
            box("wall_start", -20, 0, 1.5, 0.2, 4, 3),
        ))
        obstacle_count = {"low": 0, "medium": 4, "high": 10}[complexity]
        for index in range(obstacle_count):
            x = rng.uniform(-15, 15)
            y = rng.choice((-1.45, 1.45))
            models.append(box(
                f"corridor_feature_{index}", x, y, 0.7,
                rng.uniform(0.3, 0.8), 0.5, 1.4, "0.35 0.45 0.75 1"))
    elif scene == "hall":
        models.extend((
            box("hall_north", 0, 15, 1.5, 30, 0.2, 3),
            box("hall_south", 0, -15, 1.5, 30, 0.2, 3),
            box("hall_east", 15, 0, 1.5, 0.2, 30, 3),
            box("hall_west", -15, 0, 1.5, 0.2, 30, 3),
        ))
        obstacle_count = {"low": 2, "medium": 7, "high": 15}[complexity]
        for index in range(obstacle_count):
            models.append(box(
                f"hall_column_{index}", rng.uniform(-11, 11),
                rng.uniform(-11, 11), 1.5, 0.55, 0.55, 3,
                "0.70 0.55 0.30 1"))
    else:
        models.extend((
            box("building_north", 0, 18, 4, 36, 2, 8, "0.55 0.60 0.65 1"),
            box("building_west", -18, 2, 3, 2, 24, 6, "0.60 0.45 0.35 1"),
            box("facade_east", 18, -4, 3.5, 2, 28, 7, "0.45 0.55 0.60 1"),
        ))
        obstacle_count = {"low": 1, "medium": 5, "high": 12}[complexity]
        for index in range(obstacle_count):
            models.append(box(
                f"outdoor_feature_{index}", rng.uniform(-13, 13),
                rng.uniform(-12, 12), rng.uniform(0.5, 1.3),
                rng.uniform(0.4, 1.5), rng.uniform(0.4, 1.5),
                rng.uniform(1.0, 2.6), "0.25 0.55 0.30 1"))
    return "\n".join(models)


def world(scene, lighting, complexity, seed):
    rng = random.Random(1000 * SCENES.index(scene) + 100 * seed + len(lighting))
    diffuse = " ".join(str(value) for value in LIGHTING[lighting])
    ambient_scale = 0.25 if lighting == "dark" else 0.55
    ambient = " ".join(
        str(value * ambient_scale) for value in LIGHTING[lighting][:3]
    ) + " 1"
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="{scene}_{lighting}_{complexity}_{seed}">
    <physics name="default_physics" type="ode">
      <real_time_update_rate>1000</real_time_update_rate>
      <max_step_size>0.001</max_step_size>
    </physics>
    <scene>
      <ambient>{ambient}</ambient>
      <background>0.08 0.09 0.12 1</background>
      <shadows>true</shadows>
    </scene>
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 20 0 0 0</pose>
      <diffuse>{diffuse}</diffuse>
      <specular>{diffuse}</specular>
      <direction>-0.5 0.2 -1</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision"><geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry></collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material><ambient>0.35 0.35 0.35 1</ambient><diffuse>0.55 0.55 0.55 1</diffuse></material>
        </visual>
      </link>
    </model>
{scene_models(scene, complexity, rng)}
  </world>
</sdf>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="simulation/generated",
        help="directory receiving world files and matrix.json")
    args = parser.parse_args()
    try:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        runs = []
        for scene in SCENES:
            for lighting in LIGHTING:
                for seed, (complexity, fault, route, speed) in enumerate(
                        RUN_VARIANTS):
                    run_id = f"{scene}_{lighting}_seed{seed}"
                    run_directory = output / run_id
                    run_directory.mkdir(parents=True, exist_ok=True)
                    world_path = run_directory / "world.sdf"
                    world_path.write_text(
                        world(scene, lighting, complexity, seed),
                        encoding="utf-8")
                    run = {
                        "run_id": run_id,
                        "scene": scene,
                        "lighting": lighting,
                        "geometry_complexity": complexity,
                        "fault": fault,
                        "route": route,
                        "speed_mps": speed,
                        "seed": seed,
                        "world": f"{run_id}/world.sdf",
                        "world_sha256": hashlib.sha256(
                            world_path.read_bytes()).hexdigest(),
                    }
                    (run_directory / "run_plan.json").write_text(
                        json.dumps(run, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
                    runs.append(run)
        matrix = {
            "format_version": 1,
            "description": "3 scenes x 3 lighting states x 3 independent seeds",
            "run_count": len(runs),
            "fault_assignment": (
                "seed0 normal, seed1 visual_degraded, seed2 laser_degraded"),
            "geometry_assignment": (
                "seed0 low, seed1 medium, seed2 high"),
            "runs": runs,
        }
        matrix_path = output / "matrix.json"
        matrix_path.write_text(
            json.dumps(matrix, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(json.dumps({
            "matrix": str(matrix_path.resolve()),
            "run_count": len(runs),
            "sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        }, ensure_ascii=False, indent=2))
        return 0
    except OSError as error:
        print(f"generate_sim_worlds: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
