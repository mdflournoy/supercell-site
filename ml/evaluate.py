"""Matching detections to tracked mesocyclones (shared by training validation and the 2023 test)."""
from __future__ import annotations

import numpy as np

from frames import KM_PER_PX


def match_pairs(det_rc: np.ndarray, truth_rc: np.ndarray, max_km: float = 5.0):
    """Greedy one-to-one matching by distance (closest pairs first). Returns list of (det_i, truth_j, km)."""
    if not len(det_rc) or not len(truth_rc):
        return []
    d = np.hypot(det_rc[:, None, 0] - truth_rc[None, :, 0], det_rc[:, None, 1] - truth_rc[None, :, 1]) * KM_PER_PX
    pairs, used_d, used_t = [], set(), set()
    for k in np.argsort(d, axis=None):
        i, j = np.unravel_index(k, d.shape)
        if d[i, j] > max_km:
            break
        if i in used_d or j in used_t:
            continue
        used_d.add(i); used_t.add(j); pairs.append((int(i), int(j), float(d[i, j])))
    return pairs


def match_points(det_rc: np.ndarray, truth_rc: np.ndarray, se: np.ndarray, max_km: float = 5.0):
    """(hits, misses, false alarms) for one frame. Unmatched detections count as false alarms only
    inside the Southeast truth region (se mask on the full grid)."""
    pairs = match_pairs(det_rc, truth_rc, max_km)
    md = {p[0] for p in pairs}
    hits = len(pairs)
    misses = len(truth_rc) - hits
    false = 0
    for i, (r, c) in enumerate(det_rc):
        if i in md:
            continue
        r, c = int(r), int(c)
        if 0 <= r < se.shape[0] and 0 <= c < se.shape[1] and se[r, c]:
            false += 1
    return hits, misses, false
