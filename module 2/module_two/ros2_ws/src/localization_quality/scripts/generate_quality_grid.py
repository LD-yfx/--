#!/usr/bin/env python3
"""Aggregate fused quality into CSV/YAML/PGM/PNG/SVG/JSON grid artifacts."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import zlib


def png_chunk(chunk_type, data):
    payload = chunk_type + data
    return (
        struct.pack(">I", len(data)) + payload +
        struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    )


def write_png(path, rows):
    height = len(rows)
    width = len(rows[0])
    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    content = b"\x89PNG\r\n\x1a\n"
    content += png_chunk(
        b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    content += png_chunk(b"IDAT", zlib.compress(raw, 9))
    content += png_chunk(b"IEND", b"")
    path.write_bytes(content)


def color(quality):
    quality = max(0.0, min(1.0, quality))
    red = int(round(255 * (1.0 - quality)))
    green = int(round(190 * quality))
    return f"#{red:02x}{green:02x}28"


def read_samples(path):
    samples = []
    frames = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                x = float(row["x"])
                y = float(row["y"])
                quality = float(row["q_fused"])
                timestamp = float(row["timestamp"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (x, y, quality, timestamp)):
                continue
            if not 0.0 <= quality <= 1.0:
                raise ValueError("q_fused must be in [0, 1]")
            frame = row.get("coordinate_frame", "odom")
            if frame == "unavailable":
                continue
            frames.add(frame)
            samples.append((timestamp, x, y, quality))
    if not samples:
        raise ValueError("input CSV contains no finite spatial quality samples")
    if len(frames) != 1:
        raise ValueError(f"mixed coordinate frames are not allowed: {sorted(frames)}")
    return samples, next(iter(frames))


def aggregate(samples, resolution, padding):
    min_x = math.floor((min(item[1] for item in samples) - padding) / resolution)
    max_x = math.ceil((max(item[1] for item in samples) + padding) / resolution)
    min_y = math.floor((min(item[2] for item in samples) - padding) / resolution)
    max_y = math.ceil((max(item[2] for item in samples) + padding) / resolution)
    origin_x = min_x * resolution
    origin_y = min_y * resolution
    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)
    cells = {}
    rejected = 0
    for timestamp, x, y, quality in samples:
        grid_x = math.floor((x - origin_x) / resolution)
        grid_y = math.floor((y - origin_y) / resolution)
        if not (0 <= grid_x < width and 0 <= grid_y < height):
            rejected += 1
            continue
        key = (grid_x, grid_y)
        count, mean, m2, _ = cells.get(key, (0, 0.0, 0.0, 0.0))
        count += 1
        difference = quality - mean
        mean += difference / count
        m2 += difference * (quality - mean)
        cells[key] = (count, mean, m2, timestamp)
    return {
        "origin_x": origin_x,
        "origin_y": origin_y,
        "width": width,
        "height": height,
        "resolution": resolution,
        "cells": cells,
        "rejected": rejected,
    }


def write_csv(path, grid):
    with path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = (
            "grid_x", "grid_y", "center_x", "center_y", "mean_quality",
            "variance", "sample_count", "confidence", "last_update", "known",
        )
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        for grid_y in range(grid["height"]):
            for grid_x in range(grid["width"]):
                cell = grid["cells"].get((grid_x, grid_y))
                if cell is None:
                    mean = variance = confidence = last_update = ""
                    count = 0
                    known = 0
                else:
                    count, mean, m2, last_update = cell
                    variance = m2 / (count - 1) if count > 1 else 0.0
                    confidence = 1.0 - math.exp(-count / 5.0)
                    known = 1
                writer.writerow({
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "center_x": grid["origin_x"] + (grid_x + 0.5) * grid["resolution"],
                    "center_y": grid["origin_y"] + (grid_y + 0.5) * grid["resolution"],
                    "mean_quality": mean,
                    "variance": variance,
                    "sample_count": count,
                    "confidence": confidence,
                    "last_update": last_update,
                    "known": known,
                })


def raster(grid):
    # Image row zero is the top, whereas OccupancyGrid row zero is the bottom.
    rows = []
    for grid_y in reversed(range(grid["height"])):
        row = []
        for grid_x in range(grid["width"]):
            cell = grid["cells"].get((grid_x, grid_y))
            row.append(205 if cell is None else int(round(cell[1] * 255)))
        rows.append(row)
    return rows


def write_svg(path, grid, frame):
    scale = max(1, min(12, 900 // max(grid["width"], grid["height"])))
    width = grid["width"] * scale
    height = grid["height"] * scale
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height + 40}" viewBox="0 0 {width} {height + 40}">',
        '<rect width="100%" height="100%" fill="#d0d0d0"/>',
        f'<text x="8" y="{height + 25}" font-family="sans-serif" '
        f'font-size="14">Q(x,y), frame={frame}, high value = high quality</text>',
    ]
    for (grid_x, grid_y), (_, mean, _, _) in grid["cells"].items():
        x = grid_x * scale
        y = (grid["height"] - 1 - grid_y) * scale
        parts.append(
            f'<rect x="{x}" y="{y}" width="{scale}" height="{scale}" '
            f'fill="{color(mean)}"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("output_directory")
    parser.add_argument("--resolution", type=float, default=0.10)
    parser.add_argument("--padding", type=float, default=0.50)
    args = parser.parse_args()
    try:
        if args.resolution <= 0.0 or args.padding < 0.0:
            raise ValueError("resolution must be positive and padding non-negative")
        input_path = Path(args.input_csv)
        output = Path(args.output_directory)
        output.mkdir(parents=True, exist_ok=True)
        samples, frame = read_samples(input_path)
        grid = aggregate(samples, args.resolution, args.padding)

        csv_path = output / "quality_grid.csv"
        write_csv(csv_path, grid)
        rows = raster(grid)
        pgm_path = output / "quality_grid.pgm"
        with pgm_path.open("wb") as stream:
            stream.write(
                f"P5\n{grid['width']} {grid['height']}\n255\n".encode("ascii"))
            for row in rows:
                stream.write(bytes(row))
        png_path = output / "quality_grid.png"
        write_png(png_path, rows)
        svg_path = output / "quality_grid.svg"
        write_svg(svg_path, grid, frame)

        yaml_path = output / "quality_grid.yaml"
        yaml_path.write_text(
            "image: quality_grid.pgm\n"
            f"resolution: {grid['resolution']:.12g}\n"
            f"origin: [{grid['origin_x']:.12g}, {grid['origin_y']:.12g}, 0.0]\n"
            f"frame_id: {frame}\n"
            "unknown_value: 205\n"
            "value_semantics: higher_is_better_localization_quality\n",
            encoding="utf-8")

        artifact_names = (
            "quality_grid.csv", "quality_grid.yaml", "quality_grid.pgm",
            "quality_grid.png", "quality_grid.svg",
        )
        metadata = {
            "format_version": 1,
            "input_csv": Path(os.path.relpath(
                input_path.resolve(), start=output.resolve())).as_posix(),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "coordinate_frame": frame,
            "coordinate_semantics": (
                "global_map_quality_grid" if frame == "map"
                else "odometry_coordinate_quality_grid"
            ),
            "artifact_type": "localization_quality_grid",
            "value_semantics": "0_low_quality_1_high_quality",
            "unknown": "unvisited",
            "resolution": grid["resolution"],
            "width": grid["width"],
            "height": grid["height"],
            "origin_x": grid["origin_x"],
            "origin_y": grid["origin_y"],
            "accepted_samples": len(samples) - grid["rejected"],
            "rejected_samples": grid["rejected"],
            "known_cells": len(grid["cells"]),
            "artifacts": {},
        }
        for name in artifact_names:
            metadata["artifacts"][name] = hashlib.sha256(
                (output / name).read_bytes()).hexdigest()
        metadata_path = output / "quality_grid_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"generate_quality_grid: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
