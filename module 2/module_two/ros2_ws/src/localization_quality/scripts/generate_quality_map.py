#!/usr/bin/env python3
"""Create a dependency-free SVG trajectory map colored by fused quality."""

import csv
import hashlib
import html
import sys
from pathlib import Path
from typing import List, Tuple


WIDTH = 900
HEIGHT = 700
LEFT = 90
RIGHT = 120
TOP = 70
BOTTOM = 80


def color_for_quality(quality: float) -> str:
    quality = max(0.0, min(1.0, quality))
    if quality < 0.5:
        red = 220
        green = round(70 + 300 * quality)
    else:
        red = round(220 - 360 * (quality - 0.5))
        green = 220
    blue = 65
    return f"#{red:02x}{green:02x}{blue:02x}"


def read_points(path: Path) -> Tuple[List[Tuple[float, float, float]], str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"x", "y", "q_fused", "mode"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV contains no data rows")
    modes = {row["mode"] for row in rows}
    if len(modes) != 1:
        raise ValueError("CSV must contain exactly one mode")
    return (
        [(float(row["x"]), float(row["y"]), float(row["q_fused"])) for row in rows],
        next(iter(modes)),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} INPUT_CSV OUTPUT_SVG", file=sys.stderr)
        return 2
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    points, mode = read_points(input_path)

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax - xmin < 1e-9:
        xmin -= 0.5
        xmax += 0.5
    if ymax - ymin < 1e-9:
        ymin -= 0.5
        ymax += 0.5
    xpad = 0.05 * (xmax - xmin)
    ypad = 0.05 * (ymax - ymin)
    xmin, xmax = xmin - xpad, xmax + xpad
    ymin, ymax = ymin - ypad, ymax + ypad

    plot_width = WIDTH - LEFT - RIGHT
    plot_height = HEIGHT - TOP - BOTTOM

    def sx(value: float) -> float:
        return LEFT + (value - xmin) / (xmax - xmin) * plot_width

    def sy(value: float) -> float:
        return TOP + (ymax - value) / (ymax - ymin) * plot_height

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#222}.tick{font-size:12px}.label{font-size:15px}.title{font-size:21px;font-weight:bold}.meta{font-size:10px;fill:#555}</style>',
        f'<text x="{WIDTH / 2:.1f}" y="32" text-anchor="middle" class="title">Localization quality map — {html.escape(mode)}</text>',
    ]
    for step in range(6):
        fraction = step / 5
        px = LEFT + fraction * plot_width
        py = TOP + fraction * plot_height
        xvalue = xmin + fraction * (xmax - xmin)
        yvalue = ymax - fraction * (ymax - ymin)
        lines.extend(
            [
                f'<line x1="{px:.2f}" y1="{TOP}" x2="{px:.2f}" y2="{TOP + plot_height}" stroke="#e5e7eb"/>',
                f'<text x="{px:.2f}" y="{TOP + plot_height + 24}" text-anchor="middle" class="tick">{xvalue:.2f}</text>',
                f'<line x1="{LEFT}" y1="{py:.2f}" x2="{LEFT + plot_width}" y2="{py:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{LEFT - 12}" y="{py + 4:.2f}" text-anchor="end" class="tick">{yvalue:.2f}</text>',
            ]
        )
    lines.extend(
        [
            f'<rect x="{LEFT}" y="{TOP}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#333"/>',
            f'<text x="{LEFT + plot_width / 2:.2f}" y="{HEIGHT - 24}" text-anchor="middle" class="label">x (m)</text>',
            f'<text x="24" y="{TOP + plot_height / 2:.2f}" text-anchor="middle" transform="rotate(-90 24 {TOP + plot_height / 2:.2f})" class="label">y (m)</text>',
        ]
    )
    trajectory = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y, _ in points)
    lines.append(
        f'<polyline points="{trajectory}" fill="none" stroke="#777" stroke-width="1" opacity="0.45"/>'
    )
    for x, y, quality in points:
        lines.append(
            f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.2" fill="{color_for_quality(quality)}"/>'
        )

    colorbar_x = LEFT + plot_width + 42
    colorbar_y = TOP + 35
    colorbar_height = plot_height - 70
    for step in range(100):
        quality = 1.0 - step / 99
        y = colorbar_y + step * colorbar_height / 100
        lines.append(
            f'<rect x="{colorbar_x}" y="{y:.2f}" width="24" height="{colorbar_height / 100 + 0.5:.2f}" fill="{color_for_quality(quality)}"/>'
        )
    lines.extend(
        [
            f'<rect x="{colorbar_x}" y="{colorbar_y}" width="24" height="{colorbar_height}" fill="none" stroke="#333"/>',
            f'<text x="{colorbar_x + 12}" y="{colorbar_y - 12}" text-anchor="middle" class="label">q_fused</text>',
            f'<text x="{colorbar_x + 34}" y="{colorbar_y + 5}" class="tick">1.0</text>',
            f'<text x="{colorbar_x + 34}" y="{colorbar_y + colorbar_height / 2 + 4}" class="tick">0.5</text>',
            f'<text x="{colorbar_x + 34}" y="{colorbar_y + colorbar_height + 4}" class="tick">0.0</text>',
            f'<text x="{LEFT}" y="{HEIGHT - 5}" class="meta">source SHA-256: {sha256(input_path)}</text>',
            '</svg>',
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
