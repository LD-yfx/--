#!/usr/bin/env python3
"""Validate run manifests, split complete runs, and create supervised labels."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


SCENES = ("corridor", "hall", "outdoor")
LIGHTING = ("dark", "normal", "strong")
FAULTS = ("normal", "visual_degraded", "laser_degraded")
REQUIRED_TOPICS = (
    "rgb",
    "camera_info",
    "laser_scan",
    "point_cloud",
    "odometry",
    "visual_localization",
    "laser_localization",
    "ground_truth",
    "tf",
    "tf_static",
    "clock",
    "metadata",
)


def load_manifests(root):
    manifests = []
    for path in sorted(Path(root).glob("**/run_metadata.json")):
        with path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        manifest["_path"] = str(path.resolve())
        manifests.append(manifest)
    return manifests


def resolve_manifest_path(manifest, field):
    """Resolve a portable path relative to its run_metadata.json file."""
    value = Path(manifest[field])
    if value.is_absolute():
        return value
    manifest_path = Path(manifest["_path"])
    return (manifest_path.parent / value).resolve()


def validate_manifest(manifest):
    errors = []
    for field in (
        "run_id", "scene", "lighting", "geometry_complexity", "fault",
        "route", "speed_mps", "seed", "bag_path", "topics",
    ):
        if field not in manifest:
            errors.append(f"missing field {field}")
    if manifest.get("scene") not in SCENES:
        errors.append("invalid scene")
    if manifest.get("lighting") not in LIGHTING:
        errors.append("invalid lighting")
    if manifest.get("fault") not in FAULTS:
        errors.append("invalid fault")
    geometry = manifest.get("geometry_complexity")
    if geometry not in ("low", "medium", "high"):
        errors.append("invalid geometry_complexity")
    topics = manifest.get("topics", {})
    for topic in REQUIRED_TOPICS:
        if topic not in topics:
            errors.append(f"missing topic mapping {topic}")
    for field in ("bag_path", "world"):
        if field not in manifest:
            if field == "world":
                errors.append("missing field world")
            continue
        if Path(manifest[field]).is_absolute():
            errors.append(f"{field} must be relative to run_metadata.json")
            continue
        try:
            resolved = resolve_manifest_path(manifest, field)
            if not resolved.exists():
                errors.append(f"{field} does not exist: {manifest[field]}")
        except (KeyError, OSError, RuntimeError, ValueError):
            errors.append(f"invalid {field}")
    return errors


def command_validate(args):
    manifests = load_manifests(args.dataset_root)
    errors = {}
    run_ids = set()
    matrix_keys = set()
    for manifest in manifests:
        item_errors = validate_manifest(manifest)
        run_id = manifest.get("run_id", manifest["_path"])
        if run_id in run_ids:
            item_errors.append("duplicate run_id")
        run_ids.add(run_id)
        matrix_keys.add((
            manifest.get("scene"), manifest.get("lighting"),
            int(manifest.get("seed", -1)),
        ))
        if item_errors:
            errors[run_id] = item_errors
    expected = {
        (scene, lighting, seed)
        for scene in SCENES for lighting in LIGHTING for seed in range(3)
    }
    missing = sorted(expected - matrix_keys)
    report = {
        "manifest_count": len(manifests),
        "expected_count": 27,
        "matrix_complete": not missing and len(manifests) == 27,
        "missing_scene_lighting_seed": missing,
        "manifest_errors": errors,
        "passed": not errors and not missing and len(manifests) == 27,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


def balanced_split(manifests, seed):
    """Split whole runs 16/5/6 with scene/light/fault coverage.

    In the formal matrix, seed also selects route, speed, geometry and fault.
    Consequently a random whole-run split can accidentally reserve an entire
    degradation mode for validation or test.  This Latin-style allocation
    keeps the split causal and run-disjoint while exposing every split to all
    three seed-linked operating modes.
    """
    offset = (seed - 20260727) % 3
    by_key = {
        (item["scene"], item["lighting"], int(item["seed"])): item
        for item in manifests
    }
    validation_slots = (
        ("corridor", "dark", 0),
        ("corridor", "normal", 1),
        ("hall", "strong", 2),
        ("outdoor", "dark", 1),
        ("outdoor", "normal", 2),
    )
    test_slots = (
        ("corridor", "dark", 2),
        ("corridor", "strong", 0),
        ("hall", "dark", 1),
        ("hall", "normal", 0),
        ("outdoor", "normal", 1),
        ("outdoor", "strong", 2),
    )

    def select(slots):
        return [
            by_key[(scene, lighting, (run_seed + offset) % 3)]
            for scene, lighting, run_seed in slots
        ]

    validation = select(validation_slots)
    test = select(test_slots)
    held_out = {
        item["run_id"] for item in validation + test
    }
    train = [
        item for item in manifests if item["run_id"] not in held_out
    ]
    train.sort(key=lambda item: item["run_id"])
    validation.sort(key=lambda item: item["run_id"])
    test.sort(key=lambda item: item["run_id"])

    if (len(train), len(validation), len(test)) != (16, 5, 6):
        raise RuntimeError("stratified split did not produce 16/5/6 runs")
    return {"train": train, "validation": validation, "test": test}


def command_split(args):
    manifests = load_manifests(args.dataset_root)
    if not manifests:
        raise RuntimeError("no run_metadata.json files found")
    invalid = {
        item.get("run_id", item["_path"]): validate_manifest(item)
        for item in manifests if validate_manifest(item)
    }
    if invalid:
        raise RuntimeError(f"invalid manifests: {invalid}")
    split = balanced_split(manifests, args.seed)
    output = {
        "format_version": 1,
        "method": "complete_run_stratified_60_20_20",
        "seed": args.seed,
        "counts": {name: len(items) for name, items in split.items()},
        "runs": {
            name: [item["run_id"] for item in items]
            for name, items in split.items()
        },
    }
    run_to_split = {
        run_id: name
        for name, run_ids in output["runs"].items() for run_id in run_ids
    }
    if len(run_to_split) != len(manifests):
        raise RuntimeError("a run was duplicated or omitted by split")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def quaternion_angle(row, prefix_a, prefix_b):
    qa = [float(row[f"{prefix_a}_q{axis}"]) for axis in ("x", "y", "z", "w")]
    qb = [float(row[f"{prefix_b}_q{axis}"]) for axis in ("x", "y", "z", "w")]
    norm_a = math.sqrt(sum(value * value for value in qa))
    norm_b = math.sqrt(sum(value * value for value in qb))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return math.inf
    dot = abs(sum(a * b for a, b in zip(qa, qb)) / (norm_a * norm_b))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def pose_error(row, estimate_prefix):
    translation = math.sqrt(sum(
        (float(row[f"{estimate_prefix}_{axis}"]) - float(row[f"gt_{axis}"])) ** 2
        for axis in ("x", "y", "z")
    ))
    rotation = quaternion_angle(row, estimate_prefix, "gt")
    return translation, rotation


def quality_label(translation, rotation, translation_scale, rotation_scale):
    if not math.isfinite(translation) or not math.isfinite(rotation):
        return 0.0
    distance = math.sqrt(
        (translation / translation_scale) ** 2 +
        (rotation / rotation_scale) ** 2
    )
    return math.exp(-distance)


def oracle_from_losses(visual_loss, laser_loss, epsilon):
    if not math.isfinite(visual_loss) and not math.isfinite(laser_loss):
        return 0.5, 0.5
    if not math.isfinite(visual_loss):
        return 0.0, 1.0
    if not math.isfinite(laser_loss):
        return 1.0, 0.0
    visual_reliability = 1.0 / (visual_loss + epsilon)
    laser_reliability = 1.0 / (laser_loss + epsilon)
    total = visual_reliability + laser_reliability
    return visual_reliability / total, laser_reliability / total


def rolling_mean(values, window):
    output = []
    total = 0.0
    finite_count = 0
    queue = []
    for value in values:
        queue.append(value)
        if math.isfinite(value):
            total += value
            finite_count += 1
        if len(queue) > window:
            removed = queue.pop(0)
            if math.isfinite(removed):
                total -= removed
                finite_count -= 1
        output.append(total / finite_count if finite_count else math.inf)
    return output


def command_labels(args):
    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("pose CSV is empty")

    visual_losses, laser_losses = [], []
    visual_normalized_losses, laser_normalized_losses = [], []
    enriched = []
    skipped_invalid_pose_rows = 0
    for row in rows:
        visual_translation, visual_rotation = pose_error(row, "visual")
        laser_translation, laser_rotation = pose_error(row, "laser")
        if not all(math.isfinite(value) for value in (
                visual_translation, visual_rotation,
                laser_translation, laser_rotation)):
            skipped_invalid_pose_rows += 1
            continue
        visual_losses.append(
            visual_translation ** 2 + args.rotation_weight * visual_rotation ** 2
        )
        laser_losses.append(
            laser_translation ** 2 + args.rotation_weight * laser_rotation ** 2
        )
        visual_normalized_losses.append(
            (visual_translation / args.translation_scale) ** 2 +
            (visual_rotation / args.rotation_scale) ** 2)
        laser_normalized_losses.append(
            (laser_translation / args.translation_scale) ** 2 +
            (laser_rotation / args.rotation_scale) ** 2)
        item = dict(row)
        item.update({
            "visual_translation_error_m": visual_translation,
            "visual_rotation_error_rad": visual_rotation,
            "laser_translation_error_m": laser_translation,
            "laser_rotation_error_rad": laser_rotation,
        })
        enriched.append(item)
    if not enriched:
        raise RuntimeError("pose CSV has no rows with complete finite poses")

    visual_window = rolling_mean(visual_losses, args.window)
    laser_window = rolling_mean(laser_losses, args.window)
    visual_normalized_window = rolling_mean(
        visual_normalized_losses, args.window)
    laser_normalized_window = rolling_mean(
        laser_normalized_losses, args.window)
    for item, visual_loss, laser_loss, visual_normalized, laser_normalized in zip(
            enriched, visual_window, laser_window,
            visual_normalized_window, laser_normalized_window):
        visual_weight, laser_weight = oracle_from_losses(
            visual_loss, laser_loss, args.epsilon)
        item["visual_window_error"] = math.sqrt(max(0.0, visual_loss))
        item["laser_window_error"] = math.sqrt(max(0.0, laser_loss))
        item["q_visual_star"] = math.exp(
            -math.sqrt(max(0.0, visual_normalized)))
        item["q_laser_star"] = math.exp(
            -math.sqrt(max(0.0, laser_normalized)))
        item["w_visual_star"] = visual_weight
        item["w_laser_star"] = laser_weight

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(enriched[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(enriched)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(json.dumps({
        "rows": len(enriched),
        "output": str(output_path),
        "sha256": digest,
        "window": args.window,
        "skipped_invalid_pose_rows": skipped_invalid_pose_rows,
    }, indent=2))
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-matrix")
    validate.add_argument("dataset_root")
    validate.set_defaults(function=command_validate)

    split = subparsers.add_parser("split")
    split.add_argument("dataset_root")
    split.add_argument("output")
    split.add_argument("--seed", type=int, default=20260727)
    split.set_defaults(function=command_split)

    labels = subparsers.add_parser("labels")
    labels.add_argument("input")
    labels.add_argument("output")
    labels.add_argument("--window", type=int, default=10)
    labels.add_argument("--translation-scale", type=float, default=0.50)
    labels.add_argument("--rotation-scale", type=float, default=0.35)
    labels.add_argument("--rotation-weight", type=float, default=0.25)
    labels.add_argument("--epsilon", type=float, default=1e-6)
    labels.set_defaults(function=command_labels)
    return result


def main():
    args = parser().parse_args()
    try:
        return args.function(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"prepare_dataset: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
