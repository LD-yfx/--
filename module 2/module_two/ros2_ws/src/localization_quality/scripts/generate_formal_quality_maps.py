#!/usr/bin/env python3
"""Generate one localization-quality map per formal run and summarize them."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys


MAP_ARTIFACTS = (
    "runtime.csv",
    "quality_grid.csv",
    "quality_grid.yaml",
    "quality_grid.pgm",
    "quality_grid.png",
    "quality_grid.svg",
    "quality_grid_metadata.json",
)

SUMMARY_FIELDS = (
    "run_id",
    "split",
    "scene",
    "lighting",
    "fault",
    "geometry_complexity",
    "route",
    "seed",
    "expected_frames",
    "output_rows",
    "output_completeness",
    "spatial_samples",
    "spatial_completeness",
    "coordinate_frame",
    "known_cells",
    "coverage_area_m2",
    "mean_quality",
    "stddev_quality",
    "min_quality",
    "p05_quality",
    "p50_quality",
    "p95_quality",
    "max_quality",
    "mean_cell_confidence",
    "low_quality_cell_ratio",
    "medium_quality_cell_ratio",
    "high_quality_cell_ratio",
    "processing_p95_ms",
    "passed",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, fraction):
    if not values:
        raise ValueError("percentile input is empty")
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values):
    if not values:
        raise ValueError("quality sample collection is empty")
    return {
        "mean": statistics.fmean(values),
        "stddev": statistics.pstdev(values),
        "min": min(values),
        "p05": percentile(values, 0.05),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def relative_path(path, start):
    return Path(os.path.relpath(path.resolve(), start=start.resolve())).as_posix()


def load_manifests(dataset_root):
    manifests = []
    for path in sorted(dataset_root.glob("*/run_metadata.json")):
        manifest = read_json(path)
        manifest["_path"] = path
        manifests.append(manifest)
    if len(manifests) != 27:
        raise ValueError(
            f"formal matrix requires 27 run manifests, found {len(manifests)}")
    return manifests


def load_split(path):
    document = read_json(path)
    lookup = {}
    for subset, run_ids in document["runs"].items():
        for run_id in run_ids:
            if run_id in lookup:
                raise ValueError(f"run appears in multiple splits: {run_id}")
            lookup[run_id] = subset
    return document, lookup


def bag_artifacts(manifest):
    manifest_path = manifest["_path"]
    bag_root = (manifest_path.parent / manifest["bag_path"]).resolve()
    files = {}
    for name, expected in sorted(manifest["files"].items()):
        path = bag_root / name
        if not path.is_file():
            raise ValueError(f"{manifest['run_id']}: bag artifact missing: {name}")
        actual = sha256(path)
        if actual != expected["sha256"]:
            raise ValueError(
                f"{manifest['run_id']}: bag artifact hash mismatch: {name}")
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    return bag_root, files


def run_command(command, environment):
    result = subprocess.run(
        command,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stdout + "\n" + result.stderr).strip()
        raise RuntimeError(
            f"command exited with {result.returncode}: {' '.join(command)}\n"
            f"{detail}")


def generate_run(
        manifest, split_name, maps_root, params_file, model_path,
        grid_script, resolution, padding, environment):
    run_id = manifest["run_id"]
    bag_root, bag_files = bag_artifacts(manifest)
    final_directory = maps_root / run_id
    staging = maps_root / f".{run_id}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    runtime_csv = staging / "runtime.csv"
    try:
        run_command([
            "ros2", "run", "localization_quality", "offline_evaluator",
            str(bag_root), "normal", str(runtime_csv),
            "--ros-args",
            "--params-file", str(params_file),
            "-p", f"fusion.model_path:={model_path}",
        ], environment)
        run_command([
            sys.executable,
            str(grid_script),
            str(runtime_csv),
            str(staging),
            "--resolution", str(resolution),
            "--padding", str(padding),
        ], environment)
        if final_directory.exists():
            shutil.rmtree(final_directory)
        staging.replace(final_directory)
        print(f"generated {run_id}", flush=True)
        return {
            "run_id": run_id,
            "split": split_name,
            "bag_files": bag_files,
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def read_runtime(path):
    rows = []
    spatial = []
    processing = []
    frames = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(row)
            try:
                processing_value = float(row["processing_ms"])
                if math.isfinite(processing_value):
                    processing.append(processing_value)
                x = float(row["x"])
                y = float(row["y"])
                timestamp = float(row["timestamp"])
                quality = float(row["q_fused"])
            except (KeyError, TypeError, ValueError):
                continue
            frame = row.get("coordinate_frame", "odom")
            if (
                    all(math.isfinite(value) for value in (
                        x, y, timestamp, quality)) and
                    0.0 <= quality <= 1.0 and
                    frame != "unavailable"):
                frames.add(frame)
                spatial.append((timestamp, x, y, quality))
    if not rows or not spatial or not processing:
        raise ValueError(f"{path}: runtime evidence is incomplete")
    if len(frames) != 1:
        raise ValueError(f"{path}: coordinate frame set is {sorted(frames)}")
    return rows, spatial, processing, next(iter(frames))


def read_grid_cells(path):
    cells = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["known"] != "1":
                continue
            cells.append({
                "quality": float(row["mean_quality"]),
                "confidence": float(row["confidence"]),
                "sample_count": int(row["sample_count"]),
            })
    if not cells:
        raise ValueError(f"{path}: grid contains zero known cells")
    return cells


def summarize_run(
        manifest, split_name, run_directory, params_file, model_path,
        resolution, padding):
    runtime_path = run_directory / "runtime.csv"
    rows, spatial, processing, coordinate_frame = read_runtime(runtime_path)
    metadata = read_json(run_directory / "quality_grid_metadata.json")
    cells = read_grid_cells(run_directory / "quality_grid.csv")
    expected = int(
        manifest["topic_statistics"][
            manifest["topics"]["laser_scan"]]["message_count"])
    output_completeness = len(rows) / expected
    spatial_completeness = len(spatial) / len(rows)
    quality_values = [item[3] for item in spatial]
    confidence_values = [item["confidence"] for item in cells]
    low = sum(item["quality"] < 0.4 for item in cells)
    medium = sum(
        0.4 <= item["quality"] < 0.7 for item in cells)
    high = sum(item["quality"] >= 0.7 for item in cells)
    quality = describe(quality_values)
    artifacts = {}
    for name in MAP_ARTIFACTS:
        path = run_directory / name
        if not path.is_file():
            raise ValueError(f"{manifest['run_id']}: artifact missing: {name}")
        artifacts[name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    input_hash_matches = (
        metadata["input_sha256"] == artifacts["runtime.csv"]["sha256"])
    minimum = float(manifest.get("minimum_output_completeness", 0.95))
    passed = (
        output_completeness >= minimum and
        spatial_completeness >= minimum and
        metadata["accepted_samples"] == len(spatial) and
        metadata["known_cells"] == len(cells) and
        input_hash_matches)
    record = {
        "run_id": manifest["run_id"],
        "split": split_name,
        "scene": manifest["scene"],
        "lighting": manifest["lighting"],
        "fault": manifest["fault"],
        "geometry_complexity": manifest["geometry_complexity"],
        "route": manifest["route"],
        "seed": manifest["seed"],
        "source": {
            "run_metadata": relative_path(
                manifest["_path"], run_directory),
            "bag_path": relative_path(
                (manifest["_path"].parent / manifest["bag_path"]).resolve(),
                run_directory),
        },
        "generation": {
            "runtime_mode": "recorded_inputs",
            "quality_source": "formal_context_mlp",
            "params_sha256": sha256(params_file),
            "model_sha256": sha256(model_path),
            "resolution": resolution,
            "padding": padding,
        },
        "expected_frames": expected,
        "output_rows": len(rows),
        "output_completeness": output_completeness,
        "spatial_samples": len(spatial),
        "spatial_completeness": spatial_completeness,
        "coordinate_frame": coordinate_frame,
        "grid": {
            "width": metadata["width"],
            "height": metadata["height"],
            "origin_x": metadata["origin_x"],
            "origin_y": metadata["origin_y"],
            "known_cells": len(cells),
            "coverage_area_m2": len(cells) * resolution * resolution,
            "mean_cell_confidence": statistics.fmean(confidence_values),
            "quality_cell_ratios": {
                "low_0_0_4": low / len(cells),
                "medium_0_4_0_7": medium / len(cells),
                "high_0_7_1_0": high / len(cells),
            },
        },
        "quality": quality,
        "processing_p95_ms": percentile(processing, 0.95),
        "artifacts": artifacts,
        "passed": passed,
    }
    (run_directory / "run_summary.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return record, quality_values, cells


def summarize_group(records, quality_by_run, cells_by_run):
    qualities = []
    confidences = []
    cell_qualities = []
    for record in records:
        qualities.extend(quality_by_run[record["run_id"]])
        run_cells = cells_by_run[record["run_id"]]
        confidences.extend(item["confidence"] for item in run_cells)
        cell_qualities.extend(item["quality"] for item in run_cells)
    expected = sum(record["expected_frames"] for record in records)
    output_rows = sum(record["output_rows"] for record in records)
    spatial_samples = sum(record["spatial_samples"] for record in records)
    known_cells = sum(record["grid"]["known_cells"] for record in records)
    resolution = records[0]["generation"]["resolution"]
    return {
        "run_count": len(records),
        "expected_frames": expected,
        "output_rows": output_rows,
        "output_completeness": output_rows / expected,
        "spatial_samples": spatial_samples,
        "spatial_completeness": spatial_samples / output_rows,
        "known_cells": known_cells,
        "coverage_area_m2": known_cells * resolution * resolution,
        "mean_cell_confidence": statistics.fmean(confidences),
        "quality_cell_ratios": {
            "low_0_0_4": (
                sum(value < 0.4 for value in cell_qualities) /
                len(cell_qualities)),
            "medium_0_4_0_7": (
                sum(0.4 <= value < 0.7 for value in cell_qualities) /
                len(cell_qualities)),
            "high_0_7_1_0": (
                sum(value >= 0.7 for value in cell_qualities) /
                len(cell_qualities)),
        },
        "quality": describe(qualities),
    }


def csv_record(record):
    ratios = record["grid"]["quality_cell_ratios"]
    return {
        "run_id": record["run_id"],
        "split": record["split"],
        "scene": record["scene"],
        "lighting": record["lighting"],
        "fault": record["fault"],
        "geometry_complexity": record["geometry_complexity"],
        "route": record["route"],
        "seed": record["seed"],
        "expected_frames": record["expected_frames"],
        "output_rows": record["output_rows"],
        "output_completeness": record["output_completeness"],
        "spatial_samples": record["spatial_samples"],
        "spatial_completeness": record["spatial_completeness"],
        "coordinate_frame": record["coordinate_frame"],
        "known_cells": record["grid"]["known_cells"],
        "coverage_area_m2": record["grid"]["coverage_area_m2"],
        "mean_quality": record["quality"]["mean"],
        "stddev_quality": record["quality"]["stddev"],
        "min_quality": record["quality"]["min"],
        "p05_quality": record["quality"]["p05"],
        "p50_quality": record["quality"]["p50"],
        "p95_quality": record["quality"]["p95"],
        "max_quality": record["quality"]["max"],
        "mean_cell_confidence": record["grid"]["mean_cell_confidence"],
        "low_quality_cell_ratio": ratios["low_0_0_4"],
        "medium_quality_cell_ratio": ratios["medium_0_4_0_7"],
        "high_quality_cell_ratio": ratios["high_0_7_1_0"],
        "processing_p95_ms": record["processing_p95_ms"],
        "passed": str(record["passed"]).lower(),
    }


def write_collection(
        manifests, split_document, split_lookup, maps_root, params_file,
        model_path, resolution, padding, summary_json, summary_csv):
    records = []
    quality_by_run = {}
    cells_by_run = {}
    for manifest in sorted(manifests, key=lambda item: item["run_id"]):
        run_id = manifest["run_id"]
        if run_id not in split_lookup:
            raise ValueError(f"run is absent from split file: {run_id}")
        record, qualities, cells = summarize_run(
            manifest, split_lookup[run_id], maps_root / run_id,
            params_file, model_path, resolution, padding)
        records.append(record)
        quality_by_run[run_id] = qualities
        cells_by_run[run_id] = cells

    group_dimensions = (
        "split", "scene", "lighting", "fault", "geometry_complexity")
    groups = {}
    for dimension in group_dimensions:
        values = sorted({str(record[dimension]) for record in records})
        groups[dimension] = {
            value: summarize_group(
                [record for record in records
                 if str(record[dimension]) == value],
                quality_by_run,
                cells_by_run)
            for value in values
        }
    actual_split_counts = {
        name: sum(record["split"] == name for record in records)
        for name in ("train", "validation", "test")
    }
    expected_split_counts = {
        name: len(split_document["runs"][name])
        for name in ("train", "validation", "test")
    }
    matrix = {
        (record["scene"], record["lighting"], record["seed"])
        for record in records
    }
    checks = {
        "run_count_27": len(records) == 27,
        "scene_lighting_seed_matrix_27": len(matrix) == 27,
        "split_counts_match": actual_split_counts == expected_split_counts,
        "all_runs_passed": all(record["passed"] for record in records),
        "all_coordinate_frames_odom": (
            {record["coordinate_frame"] for record in records} == {"odom"}),
    }
    collection = {
        "format_version": 1,
        "artifact_type": "formal_quality_map_collection",
        "generation_contract": {
            "dataset_root": relative_path(
                manifests[0]["_path"].parent.parent, summary_json.parent),
            "split_file": relative_path(
                Path(split_document["_path"]), summary_json.parent),
            "split_sha256": sha256(Path(split_document["_path"])),
            "params_file": relative_path(params_file, summary_json.parent),
            "params_sha256": sha256(params_file),
            "model_path": relative_path(model_path, summary_json.parent),
            "model_sha256": sha256(model_path),
            "runtime_mode": "recorded_inputs",
            "resolution": resolution,
            "padding": padding,
            "quality_thresholds": {
                "low": "[0.0,0.4)",
                "medium": "[0.4,0.7)",
                "high": "[0.7,1.0]",
            },
            "percentile_method": "linear_interpolation",
        },
        "checks": checks,
        "passed": all(checks.values()),
        "overall": summarize_group(records, quality_by_run, cells_by_run),
        "groups": groups,
        "runs": records,
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(collection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_record(record) for record in records)
    return collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("maps_root", type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--params-file", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--resolution", type=float, default=0.10)
    parser.add_argument("--padding", type=float, default=0.50)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    try:
        dataset_root = args.dataset_root.resolve()
        maps_root = args.maps_root.resolve()
        split_path = args.split.resolve()
        params_file = args.params_file.resolve()
        model_path = args.model_path.resolve()
        summary_json = (
            args.summary_json.resolve() if args.summary_json
            else maps_root.parent / "quality_map_summary.json")
        summary_csv = (
            args.summary_csv.resolve() if args.summary_csv
            else maps_root.parent / "quality_map_summary.csv")
        if args.resolution <= 0.0 or args.padding < 0.0 or args.jobs < 1:
            raise ValueError("resolution, padding, and jobs must be positive")
        for path in (split_path, params_file, model_path):
            if not path.is_file():
                raise ValueError(f"required artifact is missing: {path}")
        manifests = load_manifests(dataset_root)
        split_document, split_lookup = load_split(split_path)
        split_document["_path"] = str(split_path)
        run_ids = {manifest["run_id"] for manifest in manifests}
        if run_ids != set(split_lookup):
            raise ValueError("formal manifests and split file contain different runs")
        maps_root.mkdir(parents=True, exist_ok=True)
        if not args.summary_only:
            environment = os.environ.copy()
            grid_script = Path(__file__).resolve().with_name(
                "generate_quality_grid.py")
            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                futures = [
                    executor.submit(
                        generate_run, manifest, split_lookup[manifest["run_id"]],
                        maps_root, params_file, model_path, grid_script,
                        args.resolution, args.padding, environment)
                    for manifest in manifests
                ]
                for future in as_completed(futures):
                    future.result()
        collection = write_collection(
            manifests, split_document, split_lookup, maps_root,
            params_file, model_path, args.resolution, args.padding,
            summary_json, summary_csv)
        print(json.dumps({
            "passed": collection["passed"],
            "run_count": len(collection["runs"]),
            "overall": collection["overall"],
            "summary_json": str(summary_json),
            "summary_csv": str(summary_csv),
        }, ensure_ascii=False, indent=2))
        return 0 if collection["passed"] else 1
    except (
            OSError, KeyError, ValueError, RuntimeError,
            subprocess.SubprocessError) as error:
        print(f"generate_formal_quality_maps: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
