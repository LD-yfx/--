#!/usr/bin/env python3
"""Run offline extraction and labels for every complete Gazebo run."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import prepare_dataset


def run(command):
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("output_directory")
    parser.add_argument("--params-file", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        manifests = prepare_dataset.load_manifests(args.dataset_root)
        if len(manifests) != 27:
            raise RuntimeError(
                f"formal dataset must contain 27 manifests, found {len(manifests)}")
        invalid = {
            item.get("run_id", item["_path"]):
                prepare_dataset.validate_manifest(item)
            for item in manifests if prepare_dataset.validate_manifest(item)
        }
        if invalid:
            raise RuntimeError(f"invalid run manifests: {invalid}")

        output = Path(args.output_directory)
        runs_directory = output / "runs"
        runs_directory.mkdir(parents=True, exist_ok=True)
        split_path = output / "dataset_split.json"
        split = prepare_dataset.balanced_split(manifests, 20260727)
        split_data = {
            "format_version": 1,
            "method": "complete_run_stratified_60_20_20",
            "seed": 20260727,
            "counts": {name: len(items) for name, items in split.items()},
            "runs": {
                name: [item["run_id"] for item in items]
                for name, items in split.items()
            },
        }
        split_path.write_text(
            json.dumps(split_data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")

        labeled_paths = []
        script_directory = Path(__file__).resolve().parent
        for index, manifest in enumerate(manifests, start=1):
            run_id = manifest["run_id"]
            run_output = runs_directory / run_id
            run_output.mkdir(parents=True, exist_ok=True)
            quality_csv = run_output / "quality_features.csv"
            joined_csv = run_output / "joined_poses.csv"
            labeled_csv = run_output / "labeled.csv"
            print(f"[{index}/27] {run_id}", flush=True)
            if args.force or not quality_csv.exists():
                bag_path = prepare_dataset.resolve_manifest_path(
                    manifest, "bag_path")
                run([
                    "ros2", "run", "localization_quality", "offline_evaluator",
                    str(bag_path), "normal", str(quality_csv),
                    "--ros-args", "--params-file",
                    str(Path(args.params_file).resolve()),
                    "-p", "fusion.mlp_enabled:=false",
                    "-p", "fusion.strategy:=rule",
                    "-p", 'fusion.model_path:=""',
                ])
            if args.force or not joined_csv.exists():
                run([
                    sys.executable,
                    str(script_directory / "extract_training_rows.py"),
                    manifest["_path"], str(quality_csv), str(joined_csv),
                ])
            if args.force or not labeled_csv.exists():
                run([
                    sys.executable,
                    str(script_directory / "prepare_dataset.py"),
                    "labels", str(joined_csv), str(labeled_csv),
                ])
            labeled_paths.append(labeled_csv)

        combined_path = output / "training_rows.csv"
        writer = None
        split_mapping = {
            run_id: name
            for name, run_ids in split_data["runs"].items()
            for run_id in run_ids
        }
        with combined_path.open("w", encoding="utf-8", newline="") as output_stream:
            for labeled_path in labeled_paths:
                with labeled_path.open(
                        "r", encoding="utf-8", newline="") as input_stream:
                    reader = csv.DictReader(input_stream)
                    if writer is None:
                        fields = list(reader.fieldnames) + ["split"]
                        writer = csv.DictWriter(
                            output_stream, fieldnames=fields, lineterminator="\n"
                        )
                        writer.writeheader()
                    for row in reader:
                        row["split"] = split_mapping[row["run_id"]]
                        writer.writerow(row)
        print(json.dumps({
            "runs": len(manifests),
            "combined_csv": str(combined_path.resolve()),
            "split": str(split_path.resolve()),
        }, ensure_ascii=False, indent=2))
        return 0
    except (OSError, KeyError, ValueError, RuntimeError,
            subprocess.SubprocessError) as error:
        print(f"build_training_dataset: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
