#!/usr/bin/env python3
"""Validate the formal 27-run localization-quality map collection."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


REQUIRED_ARTIFACTS = {
    "runtime.csv",
    "quality_grid.csv",
    "quality_grid.yaml",
    "quality_grid.pgm",
    "quality_grid.png",
    "quality_grid.svg",
    "quality_grid_metadata.json",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("maps_root", type=Path)
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--expected-runs", type=int, default=27)
    args = parser.parse_args()
    try:
        maps_root = args.maps_root.resolve()
        report = json.loads(
            args.summary_json.read_text(encoding="utf-8"))
        runs = report["runs"]
        checks = {
            "collection_passed": report["passed"] is True,
            "run_count": len(runs) == args.expected_runs,
            "unique_run_ids": (
                len({record["run_id"] for record in runs}) ==
                args.expected_runs),
            "run_reports_passed": all(record["passed"] for record in runs),
            "artifacts_complete": True,
            "artifact_hashes_match": True,
            "metadata_input_hashes_match": True,
            "run_summaries_match": True,
            "summary_csv_matches": True,
        }
        for record in runs:
            directory = maps_root / record["run_id"]
            found = {
                path.name for path in directory.iterdir() if path.is_file()
            }
            checks["artifacts_complete"] &= (
                REQUIRED_ARTIFACTS <= found and
                (directory / "run_summary.json").is_file())
            for name in REQUIRED_ARTIFACTS:
                path = directory / name
                expected = record["artifacts"].get(name, {}).get("sha256")
                checks["artifact_hashes_match"] &= (
                    path.is_file() and expected == sha256(path))
            metadata = json.loads(
                (directory / "quality_grid_metadata.json").read_text(
                    encoding="utf-8"))
            checks["metadata_input_hashes_match"] &= (
                metadata["input_sha256"] ==
                record["artifacts"]["runtime.csv"]["sha256"])
            run_summary = json.loads(
                (directory / "run_summary.json").read_text(encoding="utf-8"))
            checks["run_summaries_match"] &= run_summary == record
        with args.summary_csv.open(
                "r", encoding="utf-8", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        checks["summary_csv_matches"] = (
            len(csv_rows) == args.expected_runs and
            {row["run_id"] for row in csv_rows} ==
            {record["run_id"] for record in runs})
        result = {
            "passed": all(checks.values()),
            "checks": checks,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(f"validate_quality_map_collection: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
