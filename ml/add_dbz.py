#!/usr/bin/env python3
"""Add reflectivity context to existing per-frame detections (no model needed).

For every detection, looks up the MRMS composite reflectivity of the same frame and adds
  dbz5    max composite reflectivity within ~5 km of the detection (dBZ)
  dbz10   max composite reflectivity within ~10 km (dBZ)
The tracker can then drop weak-echo detections or tracks (ml/track.py: dbz_min, trk_dbz).

  python ml/add_dbz.py --tar tars/20230331.tar --det det/20230331_detections.csv.gz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frames import Day, extract  # noqa: E402


def dbz_maxes(ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Native 0.01° composite reflectivity (uint8 code) → max dBZ within ~5 km and ~10 km."""
    d = (ref.astype(np.float32) - 1) / 2 - 32
    d[ref == 0] = -32.0
    t = torch.from_numpy(d)[None, None]
    m5 = F.max_pool2d(t, 9, stride=1, padding=4)[0, 0].numpy()      # ±4 px ≈ ±4.4 km
    m10 = F.max_pool2d(t, 19, stride=1, padding=9)[0, 0].numpy()    # ±9 px ≈ ±10 km
    return m5, m10


def add_columns(det: pd.DataFrame, day: Day) -> pd.DataFrame:
    det = det.copy()
    det["dbz5"] = np.nan
    det["dbz10"] = np.nan
    idx = {n: i for i, n in enumerate(day.frames.name)}
    for name, g in det.groupby("frame"):
        _, _, ref = day.raw(idx[name])
        m5, m10 = dbz_maxes(ref)
        r = np.clip(g.row.to_numpy() // 2, 0, ref.shape[0] - 1)
        c = np.clip(g.col.to_numpy() // 2, 0, ref.shape[1] - 1)
        det.loc[g.index, "dbz5"] = np.round(m5[r, c], 1)
        det.loc[g.index, "dbz10"] = np.round(m10[r, c], 1)
    return det


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--det", required=True)
    a = ap.parse_args()
    day = Day(extract(Path(a.tar), Path("work-dbz")))
    det = pd.read_csv(a.det)
    out = add_columns(det, day)
    out.to_csv(a.det, index=False)
    print(a.det, len(out), "detections;", "dbz5 median", out.dbz5.median())


if __name__ == "__main__":
    main()
