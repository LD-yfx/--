"""Strict import of module-one quality grids; quality is never occupancy."""
from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class GridSpec:
    width: int
    height: int
    resolution: float
    origin_x: float = 0.0
    origin_y: float = 0.0
    yaw: float = 0.0
    frame: str = 'map'

    def __post_init__(self):
        if (not isinstance(self.width, int) or not isinstance(self.height, int)
                or self.width <= 0 or self.height <= 0 or self.width*self.height > 4_000_000):
            raise ValueError('grid dimensions must be positive integers, at most 4M cells')
        if not all(math.isfinite(v) for v in (self.resolution, self.origin_x, self.origin_y, self.yaw)):
            raise ValueError('grid geometry must be finite')
        if self.resolution <= 0 or not self.frame.strip():
            raise ValueError('positive resolution and nonempty frame required')

    @property
    def shape(self):
        return (self.height, self.width)

    def cell_center(self, x, y):
        u, v = (x+0.5)*self.resolution, (y+0.5)*self.resolution
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.origin_x+c*u-s*v, self.origin_y+s*u+c*v

    def world_cell(self, x, y):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = x-self.origin_x, y-self.origin_y
        return math.floor((c*dx+s*dy)/self.resolution), math.floor((-s*dx+c*dy)/self.resolution)


@dataclass
class QualityGrid:
    spec: GridSpec
    mean: np.ndarray
    variance: np.ndarray
    count: np.ndarray
    last_observed: np.ndarray
    known: np.ndarray
    statistics_mode: str = 'cumulative'
    source: str = ''

    def validate(self):
        if self.statistics_mode not in ('cumulative', 'windowed', 'mean_only'):
            raise ValueError('unsupported statistics mode')
        for name in ('mean', 'variance', 'count', 'last_observed', 'known'):
            value = np.asarray(getattr(self, name))
            if value.shape != self.spec.shape:
                raise ValueError(f'{name} dimensions do not match metadata')
            setattr(self, name, value)
        self.known = self.known.astype(bool)
        k = self.known
        if not np.all(np.isfinite(self.mean[k])) or np.any((self.mean[k] < 0) | (self.mean[k] > 1)):
            raise ValueError('known quality must be finite in [0,1]')
        if self.statistics_mode != 'mean_only':
            if not np.all(np.isfinite(self.variance[k])) or np.any(self.variance[k] < 0):
                raise ValueError('variance must be finite and nonnegative')
            if not np.all(np.isfinite(self.count[k])) or np.any(self.count[k] < 1) or np.any(self.count[k] != np.floor(self.count[k])):
                raise ValueError('known sample counts must be positive integers')
            if not np.all(np.isfinite(self.last_observed[k])) or np.any(self.last_observed[k] < 0):
                raise ValueError('known observation times must be finite and nonnegative')
        return self

    @classmethod
    def empty(cls, spec, statistics_mode='cumulative'):
        return cls(spec, np.zeros(spec.shape), np.zeros(spec.shape), np.zeros(spec.shape),
                   np.zeros(spec.shape), np.zeros(spec.shape, dtype=bool), statistics_mode)


def load_module_one(csv_path, metadata_path=None):
    path = Path(csv_path)
    metadata_path = Path(metadata_path) if metadata_path else path.with_name('quality_grid_metadata.json')
    meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    if meta.get('artifact_type') != 'localization_quality_grid':
        raise ValueError('metadata does not describe a localization quality grid')
    if meta.get('value_semantics') != '0_low_quality_1_high_quality':
        raise ValueError('unrecognized quality value semantics')
    spec = GridSpec(meta['width'], meta['height'], meta['resolution'],
                    meta['origin_x'], meta['origin_y'], float(meta.get('origin_yaw', 0)),
                    meta['coordinate_frame'])
    grid = QualityGrid.empty(spec)
    grid.source = str(path.resolve())
    seen = set()
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'grid_x','grid_y','mean_quality','variance','sample_count','last_update','known'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('missing module-one CSV columns')
        for row in reader:
            x, y = int(row['grid_x']), int(row['grid_y'])
            if not (0 <= x < spec.width and 0 <= y < spec.height) or (x,y) in seen:
                raise ValueError('duplicate or out-of-range grid cell')
            seen.add((x,y))
            if row['known'] not in ('0','1'):
                raise ValueError('known must be 0 or 1')
            if row['known'] == '1':
                grid.known[y,x] = True
                grid.mean[y,x] = float(row['mean_quality'])
                grid.variance[y,x] = float(row['variance'])
                grid.count[y,x] = int(row['sample_count'])
                grid.last_observed[y,x] = float(row['last_update'])
    if len(seen) != spec.width*spec.height:
        raise ValueError('CSV must represent all cells including explicit unknown cells')
    return grid.validate()
