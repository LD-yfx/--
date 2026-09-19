#!/usr/bin/env python3
"""Create a sorted SHA-256 manifest for reproducible source/data artifacts."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


EXCLUDED_DIRECTORIES = {
    "build", "install", "log", "__pycache__", ".git",
}


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("output")
    parser.add_argument(
        "--include-real-bag", action="store_true",
        help="include the multi-gigabyte my_dataset real-machine db3")
    args = parser.parse_args()
    try:
        root = Path(args.root).resolve()
        output = Path(args.output).resolve()
        entries = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.resolve() in {
                    output, output.with_suffix(".sha256")}:
                continue
            relative = path.relative_to(root)
            if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
                continue
            if any(
                    part == "_incomplete" or
                    part.startswith("formal_gazebo_rejected")
                    for part in relative.parts):
                continue
            if relative.parts and relative.parts[0] == "my_dataset":
                if not args.include_real_bag and path.suffix == ".db3":
                    continue
            entries.append({
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            })
        manifest = {
            "format_version": 1,
            "algorithm": "sha256",
            "root": ".",
            "file_count": len(entries),
            "files": entries,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        sidecar = output.with_suffix(".sha256")
        sidecar.write_text(
            f"{digest(output)}  {output.name}\n", encoding="utf-8")
        print(f"wrote {len(entries)} hashes to {output}")
        return 0
    except OSError as error:
        print(f"create_hash_manifest: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
