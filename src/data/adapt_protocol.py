"""Leakage-free per-person adaptation splits on the WHAR Arena splits (docs/lifelong_plastic_plan.md §3).

The population model always uses the harness's own split (run_whar.prepare_dataset). Inside each test person, windows
are ordered by segment (whar-datasets session id, a proxy for time):
- segment mode (most datasets): for each activity with >= 2 segments, the earliest segments form the calibration pool
  (at most all but the last segment), and the person's other segments are scored. An activity with one segment is
  scored only, so the scored label set is always the person's full label set.
- bout mode (datasets where each activity is one long segment, e.g. SAD, WISDM): the first windows of the bout form the
  calibration pool, one window is purged (it overlaps the pool), and the rest is scored.
Calibration windows are given labelled (k per class, earliest first) or unlabelled (the whole pool).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Mode rule (docs/research_plan.md §5, fixed at 02_plan): segment mode if >= 75 % of (person, activity) pairs have
# >= 2 segments, else bout mode. Shares: sad 0 %, wisdm 0 % (phase 0); wear 2 %, pamap2 18 % (03_data).
BOUT_MODE = {"sad", "wisdm", "wear", "pamap2"}


@dataclass
class PersonSplit:
    pool: np.ndarray  # positions (into the split's test arrays) of the calibration pool, in time order
    scored: np.ndarray  # positions scored for every method
    shots: dict  # k -> positions of k labelled windows per activity (earliest first)


def person_splits(subject: np.ndarray, label: np.ndarray, segment: np.ndarray, dataset_id: str,
                  ks: tuple[int, ...] = (1, 3)) -> dict[int, PersonSplit]:
    """subject, label, segment: per test window (same order as the test arrays). Returns person -> split."""
    out = {}
    kmax = max(ks)
    for p in np.unique(subject):
        idx = np.where(subject == p)[0]
        pool_parts, scored_parts = [], []
        shots = {k: [] for k in ks}
        for c in np.unique(label[idx]):
            cls = idx[label[idx] == c]
            order = cls[np.lexsort((cls, segment[cls]))]  # by segment, then window position
            segs = segment[order]
            if dataset_id in BOUT_MODE:
                n_pool = min(kmax, max(len(order) - 2, 0))
                pool, scored = order[:n_pool], order[n_pool + 1:] if n_pool else order
            else:
                uniq = np.unique(segs)
                if len(uniq) < 2:
                    pool, scored = order[:0], order
                else:
                    # earliest segments until kmax windows are pooled, keeping at least the last segment scored
                    cum = np.cumsum([np.sum(segs == u) for u in uniq])
                    n_seg = int(min(np.searchsorted(cum, kmax) + 1, len(uniq) - 1))
                    keep = np.isin(segs, uniq[:n_seg])
                    pool, scored = order[keep], order[~keep]
            pool_parts.append(pool)
            scored_parts.append(scored)
            for k in ks:
                shots[k].append(pool[:k])
        out[int(p)] = PersonSplit(pool=np.concatenate(pool_parts), scored=np.concatenate(scored_parts),
                                  shots={k: np.concatenate(v) for k, v in shots.items()})
    return out
