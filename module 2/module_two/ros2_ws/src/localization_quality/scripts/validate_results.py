#!/usr/bin/env python3
"""Validate experiment CSV invariants and the project-level acceptance targets."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, List


REQUIRED_COLUMNS = {
    "timestamp",
    "q_laser_raw",
    "q_visual_raw",
    "q_laser",
    "q_visual",
    "q_fused",
    "w_laser",
    "w_visual",
    "fusion_state",
    "mode",
    "processing_ms",
}
NUMERIC_COLUMNS = {
    "timestamp",
    "q_laser_raw",
    "q_visual_raw",
    "q_laser",
    "q_visual",
    "q_fused",
    "w_laser",
    "w_visual",
    "processing_ms",
}


def percentile(values: List[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def algorithm_sha256(rows: List[Dict[str, str]], fieldnames: List[str]) -> str:
    """Hash deterministic algorithm output while excluding machine-dependent timing."""
    columns = [name for name in fieldnames if name != "processing_ms"]
    digest = hashlib.sha256()
    digest.update((",".join(columns) + "\n").encode("utf-8"))
    for row in rows:
        digest.update((",".join(row[name] for name in columns) + "\n").encode("utf-8"))
    return digest.hexdigest()


def validate_file(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        columns = set(fieldnames)
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no data rows")

    parsed = []
    invariant_errors = 0
    previous_timestamp = -math.inf
    modes = set()
    for line_number, row in enumerate(rows, start=2):
        try:
            values = {key: float(row[key]) for key in NUMERIC_COLUMNS}
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: invalid numeric field") from error
        modes.add(row["mode"])
        if values["timestamp"] < previous_timestamp:
            invariant_errors += 1
        previous_timestamp = values["timestamp"]
        bounded = (
            values["q_laser_raw"],
            values["q_visual_raw"],
            values["q_laser"],
            values["q_visual"],
            values["q_fused"],
            values["w_laser"],
            values["w_visual"],
        )
        if any(value < -1e-9 or value > 1.0 + 1e-9 for value in bounded):
            invariant_errors += 1
        if abs(values["w_laser"] + values["w_visual"] - 1.0) > 2e-6:
            invariant_errors += 1
        expected_fused = (
            values["w_laser"] * values["q_laser"]
            + values["w_visual"] * values["q_visual"]
        )
        if abs(values["q_fused"] - expected_fused) > 2e-6:
            invariant_errors += 1
        parsed.append((row, values))

    if len(modes) != 1:
        raise ValueError(f"{path}: expected exactly one mode, got {sorted(modes)}")
    mode = next(iter(modes))
    metrics = {
        key: mean(item[1][key] for item in parsed)
        for key in (
            "q_laser_raw",
            "q_visual_raw",
            "q_laser",
            "q_visual",
            "q_fused",
            "w_laser",
            "w_visual",
            "processing_ms",
        )
    }
    metrics["processing_ms_p95"] = percentile(
        [item[1]["processing_ms"] for item in parsed], 0.95
    )

    checks = {
        "numeric_invariants": invariant_errors == 0,
        "processing_p95_le_100_ms": metrics["processing_ms_p95"] <= 100.0,
    }
    states = [item[0]["fusion_state"] for item in parsed]
    if mode == "normal":
        normal_rate = mean(state == "normal" for state in states)
        checks.update(
            {
                "mean_laser_quality_ge_0_60": metrics["q_laser"] >= 0.60,
                "mean_visual_quality_ge_0_60": metrics["q_visual"] >= 0.60,
                "normal_state_rate_ge_0_90": normal_rate >= 0.90,
            }
        )
        metrics["expected_state_rate"] = normal_rate
    elif mode == "laser_fail":
        detection_rate = mean(
            state in {"laser_degraded", "laser_invalid"} for state in states
        )
        checks.update(
            {
                "failed_laser_quality_le_0_30": metrics["q_laser"] <= 0.30,
                "healthy_visual_quality_ge_0_60": metrics["q_visual"] >= 0.60,
                "healthy_visual_weight_ge_0_85": metrics["w_visual"] >= 0.85,
                "fault_detection_rate_ge_0_95": detection_rate >= 0.95,
            }
        )
        metrics["expected_state_rate"] = detection_rate
    elif mode == "visual_fail":
        detection_rate = mean(
            state in {"visual_degraded", "visual_invalid"} for state in states
        )
        checks.update(
            {
                "failed_visual_quality_le_0_30": metrics["q_visual"] <= 0.30,
                "healthy_laser_quality_ge_0_60": metrics["q_laser"] >= 0.60,
                "healthy_laser_weight_ge_0_85": metrics["w_laser"] >= 0.85,
                "fault_detection_rate_ge_0_95": detection_rate >= 0.95,
            }
        )
        metrics["expected_state_rate"] = detection_rate
    else:
        raise ValueError(f"{path}: unsupported mode {mode!r}")

    return {
        "path": str(path),
        "file_sha256": sha256(path),
        "algorithm_sha256": algorithm_sha256(rows, fieldnames),
        "mode": mode,
        "rows": len(rows),
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    results = [validate_file(path) for path in arguments.csv]
    modes = [result["mode"] for result in results]
    required_modes = {"normal", "laser_fail", "visual_fail"}
    complete = set(modes) == required_modes and len(modes) == len(required_modes)
    summary = {
        "acceptance_profile": "undergraduate_innovation_quality_module_v1",
        "required_modes_present": complete,
        "results": results,
        "passed": complete and all(result["passed"] for result in results),
    }
    output_text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    print(output_text, end="")
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output_text, encoding="utf-8")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
