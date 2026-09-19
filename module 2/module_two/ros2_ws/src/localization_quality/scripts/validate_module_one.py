#!/usr/bin/env python3
"""Validate module-one numerical, model, runtime, and ablation acceptance gates."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def check(condition, name, actual, target):
    return {
        "name": name,
        "passed": bool(condition),
        "actual": actual,
        "target": target,
    }


def percentile(values, percentile_value):
    values = sorted(values)
    if not values:
        return math.inf
    position = (len(values) - 1) * percentile_value / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return (
        values[lower] * (upper - position) +
        values[upper] * (position - lower)
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_report_artifact(report_path, artifact):
    path = Path(artifact)
    if path.is_absolute():
        return path
    return (report_path.parent / path).resolve()


def validate_runtime(path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("runtime CSV is empty")
    invariant_errors = 0
    for row in rows:
        q_laser = float(row["q_laser"])
        q_visual = float(row["q_visual"])
        q_fused = float(row["q_fused"])
        w_laser = float(row["w_laser"])
        w_visual = float(row["w_visual"])
        values = (q_laser, q_visual, q_fused, w_laser, w_visual)
        if (
            not all(math.isfinite(value) and 0.0 <= value <= 1.0
                    for value in values)
            or abs(w_laser + w_visual - 1.0) > 1e-6
            or abs(q_fused - (
                w_laser * q_laser + w_visual * q_visual)) > 2e-6
        ):
            invariant_errors += 1
    processing = [float(row["processing_ms"]) for row in rows]
    latency = [
        float(row.get("end_to_end_latency_ms", row["processing_ms"]))
        for row in rows
    ]
    future_matches = sum(
        1 for row in rows
        if float(row.get("visual_age_sec", 0.0)) < 0.0
        or float(row.get("odom_age_sec", 0.0)) < 0.0
    )
    return {
        "rows": len(rows),
        "invariant_errors": invariant_errors,
        "processing_p95_ms": percentile(processing, 95),
        "latency_p95_ms": percentile(latency, 95),
        "future_matches": future_matches,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--runtime-csv", required=True)
    parser.add_argument("--ablation-report", required=True)
    parser.add_argument("--quality-map-report")
    parser.add_argument("--expected-sensor-frames", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        training_report_path = Path(args.training_report).resolve()
        training = json.loads(
            training_report_path.read_text(encoding="utf-8"))
        ablation = json.loads(
            Path(args.ablation_report).read_text(encoding="utf-8"))
        runtime = validate_runtime(Path(args.runtime_csv))
        test = training["metrics"]["test"]
        checks = [
            check(
                training.get("dataset_kind") == "formal",
                "formal_dataset_source", training.get("dataset_kind"), "formal"),
            check(
                test["scene_macro_f1"] >= 0.85,
                "scene_macro_f1", test["scene_macro_f1"], ">=0.85"),
            check(
                test["q_visual_negative_error_correlation"] >= 0.60,
                "q_visual_error_correlation",
                test["q_visual_negative_error_correlation"], ">=0.60"),
            check(
                test["q_laser_negative_error_correlation"] >= 0.60,
                "q_laser_error_correlation",
                test["q_laser_negative_error_correlation"], ">=0.60"),
            check(
                test["failure_auroc"] >= 0.85,
                "failure_auroc", test["failure_auroc"], ">=0.85"),
            check(
                test["weight_mae"] <= 0.10,
                "weight_mae", test["weight_mae"], "<=0.10"),
            check(
                test["improvement_over_rule"] >= 0.10,
                "improvement_over_rule",
                test["improvement_over_rule"], ">=0.10"),
            check(
                training.get("quality_contract", {}).get(
                    "rule_baseline_features") ==
                ["q_visual_raw", "q_laser_raw"],
                "rule_baseline_uses_raw_scores",
                training.get("quality_contract", {}).get(
                    "rule_baseline_features"),
                ["q_visual_raw", "q_laser_raw"]),
            check(
                runtime["invariant_errors"] == 0,
                "quality_weight_invariants",
                runtime["invariant_errors"], "0 errors"),
            check(
                runtime["future_matches"] == 0,
                "causal_sync_future_matches",
                runtime["future_matches"], "0"),
            check(
                runtime["latency_p95_ms"] <= 100.0,
                "latency_p95_ms", runtime["latency_p95_ms"], "<=100"),
        ]
        if args.expected_sensor_frames:
            completeness = runtime["rows"] / args.expected_sensor_frames
            checks.append(check(
                completeness >= 0.95, "output_completeness",
                completeness, ">=0.95"))
        model_path = resolve_report_artifact(
            training_report_path, training["model_path"])
        model_hash = sha256(model_path) if model_path.is_file() else None
        checks.append(check(
            model_hash == training.get("model_sha256"),
            "model_artifact_sha256", model_hash,
            training.get("model_sha256")))
        for artifact in training.get("dataset_inputs", []):
            artifact_path = resolve_report_artifact(
                training_report_path, artifact["path"])
            artifact_hash = (
                sha256(artifact_path) if artifact_path.is_file() else None)
            checks.append(check(
                artifact_hash == artifact["sha256"],
                f"dataset_artifact_sha256:{artifact['path']}",
                artifact_hash, artifact["sha256"]))
        split_artifact = training.get("split_file")
        if split_artifact:
            split_path = resolve_report_artifact(
                training_report_path, split_artifact["path"])
            split_hash = sha256(split_path) if split_path.is_file() else None
            checks.append(check(
                split_hash == split_artifact["sha256"],
                f"split_artifact_sha256:{split_artifact['path']}",
                split_hash, split_artifact["sha256"]))
        required_ablations = {
            "fixed_0_5", "quality_ratio_rule", "sensor_quality_mlp",
            "full_context_mlp", "lighting_geometry_context",
            "scene_geometry_context", "scene_lighting_context",
        }
        found = set(ablation["results"])
        checks.append(check(
            required_ablations <= found, "ablation_matrix",
            sorted(found), sorted(required_ablations)))
        quality_maps = None
        if args.quality_map_report:
            quality_map_path = Path(args.quality_map_report).resolve()
            quality_maps = json.loads(
                quality_map_path.read_text(encoding="utf-8"))
            overall = quality_maps["overall"]
            checks.extend([
                check(
                    quality_maps.get("artifact_type") ==
                    "formal_quality_map_collection",
                    "quality_map_artifact_type",
                    quality_maps.get("artifact_type"),
                    "formal_quality_map_collection"),
                check(
                    quality_maps.get("passed") is True,
                    "quality_map_collection",
                    quality_maps.get("passed"), True),
                check(
                    len(quality_maps["runs"]) == 27,
                    "quality_map_run_count",
                    len(quality_maps["runs"]), 27),
                check(
                    overall["output_completeness"] >= 0.95,
                    "quality_map_output_completeness",
                    overall["output_completeness"], ">=0.95"),
                check(
                    overall["spatial_completeness"] >= 0.95,
                    "quality_map_spatial_completeness",
                    overall["spatial_completeness"], ">=0.95"),
                check(
                    quality_maps["generation_contract"]["model_sha256"] ==
                    training.get("model_sha256"),
                    "quality_map_model_sha256",
                    quality_maps["generation_contract"]["model_sha256"],
                    training.get("model_sha256")),
            ])
        report = {
            "format_version": 1,
            "passed": all(item["passed"] for item in checks),
            "checks": checks,
            "runtime": runtime,
        }
        if quality_maps is not None:
            report["quality_maps"] = quality_maps["overall"]
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    except (OSError, KeyError, ValueError) as error:
        print(f"validate_module_one: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
