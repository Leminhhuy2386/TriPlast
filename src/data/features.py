"""The primary core φ: fixed per-channel value-domain features `level3+stats` (docs/research_plan.md §7.1).

Same math as phase 0 (scripts/value_probe.features; tests/test_features.py checks equality), computed in chunks of
windows so that 133-channel datasets fit in memory. Input x is (N, C, L) in harness z units.

Per channel, M = 12 levels in z units from -2.5 to 2.5:
- level: occupation (fraction of samples above each level) and level-crossing rate, of the signal and of its first
  difference scaled by the training persons' std of the difference (dx_std)  -> 4M values
- thirds: occupation over each third of the window                                                             -> 3M
then, over all channels, the window statistics mean, std, min, max                                            -> 4C
D = C (7M + 4) = 88 C. Feature groups for the label-free store G: 0 occupation, 1 crossings, 2 statistics.
"""

from __future__ import annotations

import numpy as np

LEVELS = np.linspace(-2.5, 2.5, 12)
M = len(LEVELS)


def _occupation(x: np.ndarray) -> np.ndarray:
    return (x[..., None] > LEVELS).mean(axis=2)


def _crossings(x: np.ndarray) -> np.ndarray:
    above = x[..., None] > LEVELS
    return (above[:, :, 1:] != above[:, :, :-1]).mean(axis=2)


def _features_block(x: np.ndarray, dx_std: np.ndarray) -> np.ndarray:
    n = len(x)
    dx = np.diff(x, axis=-1) / dx_std[None, :, None]
    stats = np.concatenate([x.mean(-1), x.std(-1), x.min(-1), x.max(-1)], axis=1)
    level = np.concatenate([_occupation(x), _crossings(x), _occupation(dx), _crossings(dx)], axis=-1).reshape(n, -1)
    thirds = np.array_split(np.arange(x.shape[-1]), 3)
    occ3 = np.concatenate([_occupation(x[..., idx]) for idx in thirds], axis=-1).reshape(n, -1)
    return np.concatenate([level, occ3, stats], axis=1)


def dx_std_of(x: np.ndarray) -> np.ndarray:
    """Per-channel std of the first difference over the fitting windows (x: (N, C, L))."""
    return np.diff(x.astype(np.float64), axis=-1).std(axis=(0, 2)) + 1e-8


def level3_stats(x: np.ndarray, dx_std: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """(N, C, L) -> (N, 88 C) float32, in chunks (the result equals one call on all windows)."""
    x = np.asarray(x, dtype=np.float64)
    out = [_features_block(x[i:i + chunk], dx_std) for i in range(0, len(x), chunk)]
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 88 * x.shape[1]), np.float32)


def feature_groups(channels: int) -> np.ndarray:
    """Group id per feature (0 occupation, 1 crossings, 2 statistics), in the order of level3_stats."""
    per_channel_level = [0] * M + [1] * M + [0] * M + [1] * M  # occ x, cross x, occ dx, cross dx
    groups = per_channel_level * channels + [0] * (3 * M * channels) + [2] * (4 * channels)
    return np.asarray(groups, dtype=np.int64)
