#!/usr/bin/env python3
"""Summarize and content-address a complete formal Gazebo dataset."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import prepare_dataset


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("output")
    args = parser.parse_args()
    try:
        root = Path(args.dataset_root).resolve()
        manifests = prepare_dataset.load_manifests(root)
        if len(manifests) != 27:
            raise RuntimeError(f"expected 27 manifests, found {len(manifests)}")
        topic_totals = {}
        files = []
        total_messages = 0
        for manifest in sorted(manifests, key=lambda item: item["run_id"]):
            errors = prepare_dataset.validate_manifest(manifest)
            if errors:
                raise RuntimeError(
                    f"{manifest.get('run_id')}: {', '.join(errors)}")
            for topic, statistics in manifest["topic_statistics"].items():
                count = int(statistics["message_count"])
                total_messages += count
                aggregate = topic_totals.setdefault(topic, {
                    "type": statistics["type"],
                    "total_messages": 0,
                    "minimum_per_run": count,
                    "maximum_per_run": count,
                })
                aggregate["total_messages"] += count
                aggregate["minimum_per_run"] = min(
                    aggregate["minimum_per_run"], count)
                aggregate["maximum_per_run"] = max(
                    aggregate["maximum_per_run"], count)
            run_directory = Path(manifest["_path"]).parent
            for name, metadata in sorted(manifest["files"].items()):
                files.append({
                    "run_id": manifest["run_id"],
                    "path": (run_directory / "bag" / name).relative_to(
                        root).as_posix(),
                    "bytes": int(metadata["bytes"]),
                    "sha256": metadata["sha256"],
                })
            manifest_path = Path(manifest["_path"])
            files.append({
                "run_id": manifest["run_id"],
                "path": manifest_path.relative_to(root).as_posix(),
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256(manifest_path),
            })
        content_digest = hashlib.sha256()
        for item in files:
            content_digest.update(
                f"{item['run_id']}:{item['path']}:{item['bytes']}:"
                f"{item['sha256']}\n".encode("utf-8"))
        report = {
            "format_version": 1,
            "dataset_kind": "formal",
            "matrix": "3_scene_x_3_lighting_x_3_seed",
            "run_count": len(manifests),
            "total_messages": total_messages,
            "total_hashed_bytes": sum(item["bytes"] for item in files),
            "dataset_content_sha256": content_digest.hexdigest(),
            "topic_statistics": dict(sorted(topic_totals.items())),
            "files": files,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(json.dumps({
            key: report[key] for key in (
                "run_count", "total_messages", "total_hashed_bytes",
                "dataset_content_sha256")
        }, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"summarize_formal_dataset: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
