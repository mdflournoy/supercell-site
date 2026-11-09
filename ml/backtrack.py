#!/usr/bin/env python3
"""Extend each predicted supercell backward in time by following its reflectivity cell.

The model only starts a track once the mesocyclone has a clear AzShear signal. Once a supercell
exists, we know which storm it is, so we step backward frame by frame from the first point of its
first meso and follow the same composite-reflectivity cell until the cell first appears:

  1. Take a reflectivity template (dBZ, floored at FLOOR) around the current position.
  2. In the previous frame, look for that pattern near (current position - storm motion * dt),
     within +/- SEARCH_KM; the best match (normalized cross-correlation) is the earlier position.
  3. Stop when the cell's strongest echo within ~5 km falls below stop_dbz (the cell has not formed
     yet), when the match is poor (corr < min_corr, e.g. a merger or the cell is lost), at a frame gap
     over 12 min, or after max_steps frames.

Storm motion starts from the first ~30 min of the meso track and is updated as the cell is followed.
Back-extended points get stage = "back" (detected points stage = "det"); they belong to the
supercell's first meso, so the meso track simply starts earlier.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from common import DX_REF, LAT1, LON0

KM_PER_DEG = 111.2


@dataclass(frozen=True)
class BackParams:
    stop_dbz: float = 35.0     # cell's max echo within ~5 km must be at least this to keep stepping back
    min_corr: float = 0.4      # template-match quality needed to keep following
    max_steps: int = 45        # at most ~3 h back
    half_km: float = 15.0      # template half-width
    search_km: float = 8.0     # search radius around the motion-predicted position
    floor: float = 20.0        # dBZ floor for the template (ignores light stratiform rain)


class RefFrames:
    """Composite reflectivity (dBZ) by frame time for one extracted day folder (ml/frames.extract)."""

    def __init__(self, day_dir: Path, tol: float = 90.0):
        self.tol = tol                                 # max |frame time - query time| (s)
        fr = pd.read_csv(day_dir / "frames.csv").sort_values("t")
        self.dir = Path(day_dir)
        self.t = fr.t.to_numpy(dtype=float)
        self.name = fr.name.to_numpy()

    @lru_cache(maxsize=64)
    def dbz(self, i: int) -> np.ndarray:
        with np.load(self.dir / "frames" / f"{self.name[i]}.npz") as d:
            r = d["ref"]
        out = (r.astype(np.float32) - 1) / 2 - 32
        out[r == 0] = -32.0
        return out

    def index(self, t: float) -> int:
        i = int(np.argmin(np.abs(self.t - t)))
        return i if abs(self.t[i] - t) < self.tol else -1


def rc(lat: float, lon: float) -> tuple[float, float]:
    return (LAT1 - lat) / DX_REF - 0.5, (lon - LON0) / DX_REF - 0.5


def latlon(r: float, c: float) -> tuple[float, float]:
    return LAT1 - (r + 0.5) * DX_REF, LON0 + (c + 0.5) * DX_REF


def _window(a: np.ndarray, r: int, c: int, h: int):
    if r - h < 0 or c - h < 0 or r + h + 1 > a.shape[0] or c + h + 1 > a.shape[1]:
        return None
    return a[r - h:r + h + 1, c - h:c + h + 1]


def _match(tmpl: np.ndarray, img: np.ndarray, r0: int, c0: int, s: int, h: int):
    """Best normalized cross-correlation of tmpl in img for centers within +/- s px of (r0, c0)."""
    big = _window(img, r0, c0, h + s)
    if big is None:
        return None
    tz = tmpl - tmpl.mean()
    tn = np.sqrt((tz ** 2).sum())
    if tn < 1e-6:
        return None
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(big, tmpl.shape)                      # (2s+1, 2s+1, 2h+1, 2h+1)
    wz = win - win.mean(axis=(2, 3), keepdims=True)
    num = (wz * tz).sum(axis=(2, 3))
    den = np.sqrt((wz ** 2).sum(axis=(2, 3))) * tn
    corr = np.where(den > 1e-6, num / np.maximum(den, 1e-6), -1)
    # mild preference for the motion-predicted position
    yy, xx = np.mgrid[-s:s + 1, -s:s + 1]
    corr_pen = corr - 0.002 * np.hypot(yy, xx)
    k = np.unravel_index(np.argmax(corr_pen), corr.shape)
    return r0 + k[0] - s, c0 + k[1] - s, float(corr[k])


def _motion(g: pd.DataFrame) -> tuple[float, float]:
    """Storm motion (deg lat/s, deg lon/s) from the first ~30 min of a meso track."""
    g = g.sort_values("t")
    g = g[g.t <= g.t.iloc[0] + 1800]
    if len(g) < 3 or g.t.iloc[-1] - g.t.iloc[0] < 600:
        return 0.0, 0.0
    tt = g.t.to_numpy() - g.t.iloc[0]
    return float(np.polyfit(tt, g.lat, 1)[0]), float(np.polyfit(tt, g.lon, 1)[0])


def follow_back(ref: RefFrames, t0: float, lat0: float, lon0: float, v: tuple[float, float],
                bp: BackParams) -> list[tuple[float, float, float]]:
    """Points (t, lat, lon) before t0, oldest first."""
    i = ref.index(t0)
    if i < 0:
        return []
    h = max(3, int(round(bp.half_km / (KM_PER_DEG * DX_REF))))
    s = max(2, int(round(bp.search_km / (KM_PER_DEG * DX_REF))))
    near = 4                                                               # ±4 px ≈ ±4.4 km
    vlat, vlon = v
    r, c = rc(lat0, lon0)
    out = []
    for _ in range(bp.max_steps):
        if i == 0:
            break
        dt = ref.t[i] - ref.t[i - 1]
        if dt > 720:
            break
        cur = np.maximum(ref.dbz(i), bp.floor)
        tmpl = _window(cur, int(round(r)), int(round(c)), h)
        if tmpl is None:
            break
        prev = np.maximum(ref.dbz(i - 1), bp.floor)
        pr = r + vlat * dt / DX_REF           # lat decreasing → row increasing: going back undoes motion
        pc = c - vlon * dt / DX_REF
        m = _match(tmpl, prev, int(round(pr)), int(round(pc)), s, h)
        if m is None or m[2] < bp.min_corr:
            break
        nr, nc, _ = m
        w = _window(ref.dbz(i - 1), nr, nc, near)
        if w is None or w.max() < bp.stop_dbz:
            break
        la, lo = latlon(nr, nc)
        la_prev, lo_prev = latlon(r, c)
        # update motion with the observed step (smoothed)
        vlat = 0.7 * vlat + 0.3 * (la_prev - la) / dt
        vlon = 0.7 * vlon + 0.3 * (lo_prev - lo) / dt
        r, c = nr, nc
        i -= 1
        out.append((ref.t[i], la, lo))
    return out[::-1]


def extend(pred: pd.DataFrame, ref: RefFrames, bp: BackParams) -> pd.DataFrame:
    """Add back-extended points to the first meso of every predicted supercell."""
    if not len(pred):
        return pred.assign(stage=pd.Series(dtype=str))
    pred = pred.assign(stage="det")
    add = []
    for sc, g in pred.groupby("supercell_id"):
        first = g[g.t == g.t.min()].iloc[0]
        mg = g[g.meso_id == first.meso_id]
        pts = follow_back(ref, float(first.t), float(first.lat), float(first.lon), _motion(mg), bp)
        if pts:
            row = first.to_dict()
            for t, la, lo in pts:
                add.append({**row, "t": t, "lat": la, "lon": lo, "score": np.nan, "stage": "back"})
    if add:
        a = pd.DataFrame(add)
        if "time_utc" in a:
            a["time_utc"] = pd.to_datetime(a.t, unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        pred = pd.concat([a, pred], ignore_index=True).sort_values(["supercell_id", "meso_id", "t"])
    return pred.reset_index(drop=True)
