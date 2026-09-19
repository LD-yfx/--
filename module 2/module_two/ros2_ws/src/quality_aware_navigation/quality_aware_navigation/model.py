"""Explicit, one-sided conservative costs. Scores are not probabilities."""
from dataclasses import dataclass, asdict
import math
import numpy as np
from .grid import QualityGrid


@dataclass(frozen=True)
class CostParameters:
    beta: float = 1.0
    gamma: float = 1.0
    sample_scale: float = 5.0
    age_scale: float = 30.0
    unknown_risk: float = 0.7
    max_cost: int = 200
    mean_only_floor: float = 0.7

    def __post_init__(self):
        if not all(math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError('parameters must be finite')
        if self.beta < 0 or self.gamma <= 0 or self.sample_scale <= 0 or self.age_scale <= 0:
            raise ValueError('invalid model parameter domain')
        if not 0 <= self.unknown_risk <= 1 or not 0 <= self.mean_only_floor <= 1:
            raise ValueError('risk scores must be in [0,1]')
        if not isinstance(self.max_cost, int) or not 0 <= self.max_cost <= 252:
            raise ValueError('max_cost must be an integer in [0,252]')


class CostModel:
    MODES = ('geometric', 'linear', 'nonlinear', 'full', 'no_variance', 'no_count', 'no_age')

    def __init__(self, parameters=None, mode='full'):
        self.parameters = parameters or CostParameters()
        if mode not in self.MODES:
            raise ValueError('unknown cost model mode')
        self.mode = mode

    def risks(self, grid: QualityGrid, as_of: float):
        grid.validate()
        if not math.isfinite(as_of) or as_of < 0:
            raise ValueError('as_of must be finite and use the source clock domain')
        p, k = self.parameters, grid.known
        risk = np.full(grid.spec.shape, p.unknown_risk, dtype=float)
        if self.mode == 'geometric':
            return np.zeros(grid.spec.shape)
        if not k.any():
            return risk
        mu = grid.mean[k]
        if self.mode in ('linear', 'nonlinear'):
            risk[k] = (1-mu)**(1 if self.mode == 'linear' else p.gamma)
            return risk
        if grid.statistics_mode == 'mean_only':
            # No invented counts, variances or observation ages.
            risk[k] = np.maximum((1-mu)**p.gamma, p.mean_only_floor)
            return risk
        observed = grid.last_observed[k]
        if np.any(observed > as_of + 1e-6):
            raise ValueError('future cell observations: mixed clocks or invalid snapshot')
        spread = 0 if self.mode == 'no_variance' else p.beta*np.sqrt(grid.variance[k])
        conservative = np.clip(mu-spread, 0, 1)
        base = (1-conservative)**p.gamma
        sample_support = 1.0 if self.mode == 'no_count' else -np.expm1(-grid.count[k]/p.sample_scale)
        freshness = 1.0 if self.mode == 'no_age' else np.exp(-np.maximum(0, as_of-observed)/p.age_scale)
        reliability = sample_support*freshness
        risk[k] = base + (1-reliability)*np.maximum(0, p.unknown_risk-base)
        return np.clip(risk, 0, 1)

    def costs(self, grid, as_of):
        # Match non-negative C++ lround (Python round uses ties-to-even).
        return np.floor(self.risks(grid, as_of)*self.parameters.max_cost + 0.5).astype(np.uint8)
