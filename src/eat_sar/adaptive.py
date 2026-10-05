"""Two forms of T(e): piecewise constant and linear/polynomial fit."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PiecewiseT:
    bins: list[float]   # boundaries, length = len(thresholds)+1
    thresholds: list[float]

    def __call__(self, e: float | np.ndarray) -> np.ndarray:
        e = np.atleast_1d(np.asarray(e, dtype=float))
        out = np.empty_like(e)
        for i, T in enumerate(self.thresholds):
            mask = (e >= self.bins[i]) & (e < self.bins[i + 1])
            out[mask] = T
        # 边界: 超出最右边界的归到最后一段
        out[e >= self.bins[-1]] = self.thresholds[-1]
        return out


@dataclass
class LinearT:
    a: float
    b: float
    clip_min: float = 0.05
    clip_max: float = 0.90

    def __call__(self, e: float | np.ndarray) -> np.ndarray:
        e = np.atleast_1d(np.asarray(e, dtype=float))
        return np.clip(self.a * e + self.b, self.clip_min, self.clip_max)


def fit_piecewise_from_scan(scan_df, bin_col: str = "env_bin",
                              metric: str = "f1") -> PiecewiseT:
    """
    Given the threshold-scan CSV (with rows for ALL + each env_bin), return a
    PiecewiseT whose thresholds are the argmax-of-metric thresholds per bin.
    Bin boundaries come from default config; here we just store labels->T.
    """
    # Use only non-ALL rows
    df = scan_df[scan_df[bin_col] != "ALL"]
    best = (df.sort_values(metric, ascending=False)
              .groupby(bin_col, as_index=False).head(1))
    # Need an ORDER for bins; assume caller provides the right order
    return best.set_index(bin_col)["threshold"].to_dict()


def fit_linear(env_values: np.ndarray, best_thresholds: np.ndarray) -> LinearT:
    """Least-squares linear fit T = a*e + b."""
    e = np.asarray(env_values, dtype=float)
    t = np.asarray(best_thresholds, dtype=float)
    A = np.stack([e, np.ones_like(e)], axis=1)
    a, b = np.linalg.lstsq(A, t, rcond=None)[0]
    return LinearT(a=float(a), b=float(b))
