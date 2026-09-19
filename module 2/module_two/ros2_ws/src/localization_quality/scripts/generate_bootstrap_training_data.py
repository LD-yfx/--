#!/usr/bin/env python3
"""Create deterministic synthetic rows for tests and a non-final bootstrap MLP."""

import argparse
import csv
import math
from pathlib import Path
import random

import prepare_dataset


def clip(value):
    return max(0.0, min(1.0, value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--frames-per-run", type=int, default=50)
    args = parser.parse_args()
    output = []
    geometry_values = ("low", "medium", "high")
    faults = ("normal", "visual_degraded", "laser_degraded")
    light_values = {
        "dark": (0.14, 0.22, 0.55, 0.0),
        "normal": (0.50, 0.95, 0.01, 0.01),
        "strong": (0.90, 0.55, 0.0, 0.35),
    }
    scene_values = {
        "corridor": (0.28, 0.90, 0.30, (0.92, 0.05, 0.03)),
        "hall": (0.62, 0.25, 0.62, (0.05, 0.90, 0.05)),
        "outdoor": (0.92, 0.18, 0.42, (0.03, 0.07, 0.90)),
    }
    for scene in prepare_dataset.SCENES:
        for lighting in prepare_dataset.LIGHTING:
            for seed in range(3):
                rng = random.Random(
                    10000 * prepare_dataset.SCENES.index(scene) +
                    1000 * prepare_dataset.LIGHTING.index(lighting) + seed)
                geometry_label = geometry_values[seed]
                fault = faults[seed]
                geometry = (0.18, 0.52, 0.86)[seed]
                openness, anisotropy, base_complexity, probabilities = (
                    scene_values[scene])
                illumination, light_quality, under, over = light_values[lighting]
                run_id = f"{scene}_{lighting}_seed{seed}"
                for frame in range(args.frames_per_run):
                    noise = lambda scale: rng.uniform(-scale, scale)
                    feature_match = clip(
                        0.82 * light_quality -
                        (0.55 if fault == "visual_degraded" else 0.0) +
                        noise(0.04))
                    edge = clip(
                        0.08 + 0.10 * base_complexity + noise(0.02))
                    features = clip(
                        0.78 * light_quality -
                        (0.5 if fault == "visual_degraded" else 0.0) +
                        noise(0.04))
                    q_visual = clip(
                        0.58 + 0.26 * light_quality + 0.12 * feature_match -
                        0.58 * (fault == "visual_degraded") + noise(0.025))
                    corridor_penalty = 0.16 if scene == "corridor" else 0.0
                    q_laser = clip(
                        0.82 + 0.12 * geometry - corridor_penalty -
                        0.65 * (fault == "laser_degraded") + noise(0.025))

                    visual_error = (
                        0.03 + 0.36 * (1.0 - q_visual) +
                        (0.42 if lighting == "dark" else 0.0) +
                        (0.22 if lighting == "strong" else 0.0) +
                        (0.42 if scene == "outdoor" else 0.0) +
                        (0.65 if fault == "visual_degraded" else 0.0) +
                        abs(noise(0.03)))
                    laser_error = (
                        0.03 + 0.36 * (1.0 - q_laser) +
                        (0.48 if geometry_label == "low" else 0.0) +
                        (0.38 if scene == "corridor" else 0.0) +
                        (0.70 if fault == "laser_degraded" else 0.0) +
                        abs(noise(0.03)))
                    visual_rotation = 0.45 * visual_error
                    laser_rotation = 0.45 * laser_error
                    visual_loss = (
                        visual_error ** 2 + 0.25 * visual_rotation ** 2)
                    laser_loss = laser_error ** 2 + 0.25 * laser_rotation ** 2
                    visual_reliability = 1.0 / (visual_loss + 1e-6)
                    laser_reliability = 1.0 / (laser_loss + 1e-6)
                    total = visual_reliability + laser_reliability
                    output.append({
                        "run_id": run_id,
                        "scene": scene,
                        "lighting": lighting,
                        "geometry_label": geometry_label,
                        "fault": fault,
                        "timestamp": frame * 0.1,
                        "illumination_level": clip(illumination + noise(0.03)),
                        "illumination_quality": clip(light_quality + noise(0.03)),
                        "underexposed_ratio": clip(under + noise(0.02)),
                        "overexposed_ratio": clip(over + noise(0.02)),
                        "edge_density": edge,
                        "trackable_features_norm": features,
                        "feature_match_rate": feature_match,
                        "openness": clip(openness + noise(0.04)),
                        "direction_entropy": clip(
                            0.72 - 0.5 * anisotropy + 0.2 * geometry +
                            noise(0.03)),
                        "directional_anisotropy": clip(
                            anisotropy + noise(0.03)),
                        "geometry_complexity": clip(
                            0.55 * geometry + 0.45 * base_complexity +
                            noise(0.03)),
                        "coverage_2d": clip(
                            0.78 + 0.18 * openness + noise(0.03)),
                        "linearity": clip(anisotropy + noise(0.02)),
                        "planarity": clip(
                            1.0 - anisotropy + noise(0.02)),
                        "q_visual": q_visual,
                        "q_laser": q_laser,
                        "q_visual_raw": q_visual,
                        "q_laser_raw": q_laser,
                        "laser_valid_rate": clip(
                            0.96 - 0.65 * (fault == "laser_degraded") +
                            noise(0.02)),
                        "laser_close_ratio": clip(
                            0.05 + 0.80 * (fault == "laser_degraded") +
                            noise(0.02)),
                        "p_corridor": probabilities[0],
                        "p_hall": probabilities[1],
                        "p_outdoor": probabilities[2],
                        "visual_valid": 1,
                        "laser_valid": 1,
                        "w_visual_star": visual_reliability / total,
                        "w_laser_star": laser_reliability / total,
                        "q_visual_star": math.exp(-visual_error / 0.5),
                        "q_laser_star": math.exp(-laser_error / 0.5),
                        "visual_translation_error_m": visual_error,
                        "visual_rotation_error_rad": visual_rotation,
                        "laser_translation_error_m": laser_error,
                        "laser_rotation_error_rad": laser_rotation,
                    })
    split = prepare_dataset.balanced_split([
        {
            "run_id": f"{scene}_{lighting}_seed{seed}",
            "scene": scene, "lighting": lighting, "seed": seed,
        }
        for scene in prepare_dataset.SCENES
        for lighting in prepare_dataset.LIGHTING
        for seed in range(3)
    ], 20260727)
    mapping = {
        item["run_id"]: name
        for name, items in split.items() for item in items
    }
    for row in output:
        row["split"] = mapping[row["run_id"]]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(output[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {len(output)} bootstrap rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
