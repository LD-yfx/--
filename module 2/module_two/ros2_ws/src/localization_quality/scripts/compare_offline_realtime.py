#!/usr/bin/env python3
"""Compare causal offline and realtime quality outputs at common timestamps."""

import argparse
import csv
import json
import math
from pathlib import Path
import sys


QUALITY_COLUMNS = (
    "q_laser_raw",
    "q_visual_raw",
    "q_laser",
    "q_visual",
    "q_fused",
    "w_laser",
    "w_visual",
    "p_corridor",
    "p_hall",
    "p_outdoor",
    "illumination_quality",
    "geometry_complexity",
    "openness",
)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def read_timestamped(path):
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"{path}: no rows")
    missing = sorted({"timestamp", *QUALITY_COLUMNS} - set(rows[0]))
    if missing:
        raise RuntimeError(f"{path}: missing columns: {', '.join(missing)}")
    return {round(float(row["timestamp"]), 6): row for row in rows}


def finite_column(rows, column):
    values = []
    for row in rows.values():
        if column not in row:
            continue
        value = float(row[column])
        if math.isfinite(value):
            values.append(value)
    return values


def compare(offline_path, realtime_path, expected_frames=None):
    offline = read_timestamped(offline_path)
    realtime = read_timestamped(realtime_path)
    timestamps = sorted(set(offline) & set(realtime))
    if not timestamps:
        raise RuntimeError("offline and realtime outputs have no common timestamps")
    metrics = {}
    for column in QUALITY_COLUMNS:
        differences = [
            abs(float(offline[stamp][column]) - float(realtime[stamp][column]))
            for stamp in timestamps
        ]
        metrics[column] = {
            "mean_absolute_error": sum(differences) / len(differences),
            "p95_absolute_error": percentile(differences, 0.95),
            "max_absolute_error": max(differences),
        }
    reference_count = expected_frames or len(offline)
    completeness = len(realtime) / reference_count
    performance = {}
    for column in ("processing_ms", "end_to_end_latency_ms"):
        values = finite_column(realtime, column)
        if values:
            performance[column] = {
                "mean": sum(values) / len(values),
                "p95": percentile(values, 0.95),
                "max": max(values),
            }
    checks = {
        "realtime_output_completeness_ge_0_95": completeness >= 0.95,
        "matched_realtime_rows_ge_0_99": (
            len(timestamps) / len(realtime) >= 0.99),
        "q_fused_p95_absolute_error_le_0_01": (
            metrics["q_fused"]["p95_absolute_error"] <= 0.01),
        "weight_p95_absolute_error_le_0_01": max(
            metrics["w_laser"]["p95_absolute_error"],
            metrics["w_visual"]["p95_absolute_error"]) <= 0.01,
    }
    if "end_to_end_latency_ms" in performance:
        checks["end_to_end_latency_p95_le_100_ms"] = (
            performance["end_to_end_latency_ms"]["p95"] <= 100.0)
    return {
        "format_version": 1,
        "sync_semantics": "causal_past",
        "offline_rows": len(offline),
        "realtime_rows": len(realtime),
        "matched_rows": len(timestamps),
        "expected_sensor_frames": reference_count,
        "realtime_output_completeness": completeness,
        "realtime_performance": performance,
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("offline_csv")
    parser.add_argument("realtime_csv")
    parser.add_argument("--expected-sensor-frames", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        report = compare(
            args.offline_csv, args.realtime_csv,
            args.expected_sensor_frames)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"compare_offline_realtime: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
