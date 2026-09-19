#!/usr/bin/env python3

import csv
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIRECTORY))

import prepare_dataset
import collect_sim_dataset
import train_context_mlp
import compare_offline_realtime


class DatasetPipelineTest(unittest.TestCase):
    def test_label_generation_drops_incomplete_pose_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            source = temporary / "joined.csv"
            output = temporary / "labeled.csv"
            fields = [
                f"{prefix}_{suffix}"
                for prefix in ("visual", "laser", "gt")
                for suffix in (
                    "x", "y", "z", "qx", "qy", "qz", "qw")
            ]
            valid = {field: "0" for field in fields}
            for prefix in ("visual", "laser", "gt"):
                valid[f"{prefix}_qw"] = "1"
            invalid = dict(valid)
            invalid["visual_x"] = "nan"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows((valid, invalid))
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIRECTORY / "prepare_dataset.py"),
                    "labels", str(source), str(output),
                ],
                capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            with output.open("r", encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            self.assertEqual(
                json.loads(result.stdout)["skipped_invalid_pose_rows"], 1)

    def test_bag_contract_enforces_native_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            bag = Path(temporary)
            database_path = bag / "bag_0.db3"
            with sqlite3.connect(database_path) as database:
                database.execute(
                    "CREATE TABLE topics("
                    "id INTEGER PRIMARY KEY, name TEXT, type TEXT)")
                database.execute(
                    "CREATE TABLE messages("
                    "id INTEGER PRIMARY KEY, topic_id INTEGER)")
                message_id = 1
                for topic_id, (name, message_type) in enumerate(
                        collect_sim_dataset.TOPIC_TYPES.items(), start=1):
                    database.execute(
                        "INSERT INTO topics VALUES (?, ?, ?)",
                        (topic_id, name, message_type))
                    rate = collect_sim_dataset.EXPECTED_RATES_HZ.get(name)
                    count = max(1, int(rate * 0.95)) if rate else 1
                    database.executemany(
                        "INSERT INTO messages VALUES (?, ?)",
                        ((message_id + offset, topic_id)
                         for offset in range(count)))
                    message_id += count
            statistics = collect_sim_dataset.validate_bag_topics(bag, 4.0)
            self.assertEqual(
                set(statistics), set(collect_sim_dataset.TOPIC_TYPES))

            with sqlite3.connect(database_path) as database:
                image_id = database.execute(
                    "SELECT id FROM topics WHERE name = ?",
                    ("/camera/image_raw",)).fetchone()[0]
                database.execute(
                    "DELETE FROM messages WHERE topic_id = ?",
                    (image_id,))
            with self.assertRaisesRegex(RuntimeError, "95% native-rate"):
                collect_sim_dataset.validate_bag_topics(bag, 4.0)

    def test_complete_run_split_is_disjoint(self):
        manifests = []
        for scene in prepare_dataset.SCENES:
            for lighting in prepare_dataset.LIGHTING:
                for seed in range(3):
                    manifests.append({
                        "run_id": f"{scene}_{lighting}_{seed}",
                        "scene": scene,
                        "lighting": lighting,
                        "seed": seed,
                    })
        split = prepare_dataset.balanced_split(manifests, 7)
        self.assertEqual(
            {name: len(items) for name, items in split.items()},
            {"train": 16, "validation": 5, "test": 6})
        sets = [
            {item["run_id"] for item in split[name]}
            for name in ("train", "validation", "test")
        ]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        for subset in split.values():
            self.assertEqual(
                {item["scene"] for item in subset},
                set(prepare_dataset.SCENES))
            self.assertEqual(
                {item["lighting"] for item in subset},
                set(prepare_dataset.LIGHTING))
            self.assertEqual(
                {int(item["seed"]) for item in subset}, {0, 1, 2})

    def test_numpy_mlp_learns_softmax_target(self):
        rng = np.random.RandomState(3)
        features = rng.normal(size=(200, 2))
        labels = (features[:, 0] > features[:, 1]).astype(int)
        targets = np.zeros((len(labels), 2))
        targets[np.arange(len(labels)), labels] = 1.0
        model = train_context_mlp.Mlp((2, 8, 4, 2), rng)
        before = np.mean(np.argmax(model.forward(features)[0], axis=1) == labels)
        train_context_mlp.train(model, features, targets, 80, 32, 0.01, 9)
        after = np.mean(np.argmax(model.forward(features)[0], axis=1) == labels)
        self.assertGreater(after, 0.95)
        self.assertGreater(after, before)

    def test_grid_export_has_all_required_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            source = temporary / "quality.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=(
                        "timestamp", "x", "y", "q_fused",
                        "coordinate_frame"))
                writer.writeheader()
                writer.writerows((
                    {
                        "timestamp": 1, "x": 0.1, "y": 0.1,
                        "q_fused": 0.2, "coordinate_frame": "odom",
                    },
                    {
                        "timestamp": 2, "x": 0.1, "y": 0.1,
                        "q_fused": 0.8, "coordinate_frame": "odom",
                    },
                ))
            output = temporary / "grid"
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIRECTORY / "generate_quality_grid.py"),
                    str(source), str(output), "--resolution", "1.0",
                    "--padding", "0",
                ],
                capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in (
                "quality_grid.csv", "quality_grid.yaml", "quality_grid.pgm",
                "quality_grid.png", "quality_grid.svg",
                "quality_grid_metadata.json",
            ):
                self.assertTrue((output / name).is_file(), name)
            metadata = json.loads(
                (output / "quality_grid_metadata.json").read_text(
                    encoding="utf-8"))
            self.assertEqual(
                metadata["coordinate_semantics"],
                "odometry_coordinate_quality_grid")
            with (output / "quality_grid.csv").open(
                    "r", encoding="utf-8", newline="") as stream:
                known = [
                    row for row in csv.DictReader(stream)
                    if row["known"] == "1"
                ]
            self.assertEqual(len(known), 1)
            self.assertAlmostEqual(float(known[0]["mean_quality"]), 0.5)
            self.assertAlmostEqual(float(known[0]["variance"]), 0.18)

    def test_offline_realtime_consistency_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            fields = ("timestamp",) + compare_offline_realtime.QUALITY_COLUMNS
            offline = temporary / "offline.csv"
            realtime = temporary / "realtime.csv"
            rows = []
            for index in range(20):
                row = {name: "0.5" for name in fields}
                row["timestamp"] = str(index)
                rows.append(row)
            for path, selected in ((offline, rows), (realtime, rows[1:])):
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(selected)
            report = compare_offline_realtime.compare(
                offline, realtime, expected_frames=20)
            self.assertTrue(report["passed"])
            self.assertEqual(report["matched_rows"], 19)
            self.assertAlmostEqual(
                report["metrics"]["q_fused"]["p95_absolute_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
