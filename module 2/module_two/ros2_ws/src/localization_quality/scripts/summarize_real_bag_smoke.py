#!/usr/bin/env python3
"""Summarize real-bag interface and runtime evidence."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize(path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    numeric = (
        "q_visual_raw", "q_laser_raw", "q_visual", "q_laser",
        "q_fused", "processing_ms",
    )
    metrics = {
        name: sum(float(row[name]) for row in rows) / len(rows)
        for name in numeric
    }
    metrics["processing_p95_ms"] = percentile(
        [float(row["processing_ms"]) for row in rows], 0.95)
    states = Counter(row["fusion_state"] for row in rows)
    strategies = Counter(row.get("strategy", "") for row in rows)
    domain_guarded_rows = sum(
        row.get("calibration_domain_guard", "").strip().lower()
        in {"1", "true", "yes"}
        for row in rows)
    return {
        "rows": len(rows),
        "algorithm_sha256": algorithm_sha256(rows),
        "metrics": metrics,
        "fusion_states": dict(sorted(states.items())),
        "strategies": dict(sorted(strategies.items())),
        "calibration_domain_guarded_rows": domain_guarded_rows,
    }


def algorithm_sha256(rows):
    fields = [
        name for name in rows[0]
        if name not in {"processing_ms", "end_to_end_latency_ms"}
    ]
    digest = hashlib.sha256()
    digest.update((",".join(fields) + "\n").encode("utf-8"))
    for row in rows:
        digest.update(
            (",".join(row[name] for name in fields) + "\n").encode("utf-8"))
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-root", required=True, type=Path)
    parser.add_argument("--formal-model-csv", required=True, type=Path)
    parser.add_argument("--rule-csv", required=True, type=Path)
    parser.add_argument("--expected-frames", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        formal = summarize(args.formal_model_csv)
        rule = summarize(args.rule_csv)
        bag_files = []
        for path in sorted(args.bag_root.iterdir()):
            if path.is_file():
                bag_files.append({
                    "path": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                })
        interface_passed = (
            formal["rows"] == args.expected_frames and
            rule["rows"] == args.expected_frames and
            formal["metrics"]["processing_p95_ms"] <= 100.0 and
            rule["metrics"]["processing_p95_ms"] <= 100.0)
        visual_quality_recovered = (
            formal["metrics"]["q_visual"] >= 0.60 and
            formal["metrics"]["q_visual"] >=
            formal["metrics"]["q_visual_raw"] - 0.05)
        domain_guard_coverage = (
            formal["calibration_domain_guarded_rows"] /
            formal["rows"] >= 0.95)
        no_spurious_visual_failure = (
            formal["fusion_states"].get("visual_degraded", 0) == 0 and
            formal["fusion_states"].get("both_degraded", 0) == 0)
        passed = (
            interface_passed and visual_quality_recovered and
            domain_guard_coverage and no_spurious_visual_failure)
        report = {
            "format_version": 2,
            "dataset_kind": "real_machine_bag",
            "purpose": "interface_and_runtime_evidence",
            "validated_capabilities": [
                "message_compatibility",
                "output_completeness",
                "processing_latency",
                "simulation_to_hardware_domain_shift_protection",
            ],
            "bag_files": bag_files,
            "expected_laser_frames": args.expected_frames,
            "formal_gazebo_model": formal,
            "rule_fallback": rule,
            "checks": {
                "interface_output_complete": interface_passed,
                "processing_p95_le_100_ms": (
                    formal["metrics"]["processing_p95_ms"] <= 100.0 and
                    rule["metrics"]["processing_p95_ms"] <= 100.0),
                "visual_quality_recovered_ge_0_60": visual_quality_recovered,
                "domain_guard_coverage_ge_0_95": domain_guard_coverage,
                "no_spurious_visual_failure": no_spurious_visual_failure,
            },
            "deployment_decision": {
                "hardware_strategy": "domain_guard_rule_fallback",
                "gazebo_model_role": "guarded_simulation_reference",
                "basis": (
                    "When healthy real-image evidence conflicts with the "
                    "Gazebo-only calibrator, the evaluator uses sensor-native "
                    "scores and rule fusion for that frame."
                ),
            },
            "passed": passed,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if interface_passed else 1
    except (OSError, KeyError, ValueError) as error:
        print(f"summarize_real_bag_smoke: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
