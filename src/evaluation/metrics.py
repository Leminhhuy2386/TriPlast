"""Metrics, frozen at 03_data (docs/research_plan.md §9).

macro_f1 reimplements the WHAR Arena harness definition (whar_arena.training._macro_f1) in numpy so that every metric
can be recomputed from a prediction file without the harness: the mean over *all* num_classes classes of the per-class
F1, where a class with 2·TP + FP + FN = 0 (absent and never predicted) counts 0. The harness computes in float32;
this module computes in float64, so the two agree to ~1e-7.
"""

from __future__ import annotations

import numpy as np

HARM_THRESHOLD = 0.5  # F1 points: a person is harmed if the method's gain over base is below -0.5


def confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_pred.shape}")
    if len(y_true) and (min(y_true.min(), y_pred.min()) < 0 or max(y_true.max(), y_pred.max()) >= num_classes):
        raise ValueError("labels outside [0, num_classes)")
    return np.bincount(y_true * num_classes + y_pred, minlength=num_classes * num_classes).reshape(num_classes,
                                                                                               num_classes)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Harness macro-F1 in [0, 1] (0 for an empty set, as the harness)."""
    if len(y_true) == 0:
        return 0.0
    c = confusion(y_true, y_pred, num_classes).astype(np.float64)
    tp = np.diag(c)
    denom = 2 * tp + (c.sum(0) - tp) + (c.sum(1) - tp)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def gain(method_f1: float, base_f1: float) -> float:
    """Gain in F1 points (both arguments in [0, 1])."""
    return 100.0 * (method_f1 - base_f1)


def harmed(gain_points: float) -> bool:
    return gain_points < -HARM_THRESHOLD
