"""Stage 3: link per-frame meso detections into meso tracks, and group meso tracks into supercells.

Meso linking (frame to frame, ~4 min apart)
  * each live track predicts its next position from its recent motion (last ~12 min);
  * detections are matched to predictions greedily by distance within GATE_KM;
  * a track may go unmatched for up to MAX_GAP frames before it ends;
  * a track can only START on a detection with score >= start_thr, but can CONTINUE on weaker
    detections (score >= cont_thr) — hysteresis that keeps tracks whole through weak periods;
  * tracks with fewer than MIN_LEN detections are dropped.

Supercell grouping (cyclic mesocyclogenesis)
  A new meso track joins the supercell of an existing meso track if, when it starts, that track is
  alive or ended at most JOIN_MIN minutes earlier, and the new meso is within JOIN_KM of it (its
  current or last position). Otherwise it starts a new supercell. Supercells are lettered A, B, …
  by start time, mesos numbered 1, 2, … by start time within each supercell (as in tracks/).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

KM_LAT = 111.2


@dataclass
class Params:
    start_thr: float = 0.3
    cont_thr: float = 0.15
    gate_km: float = 8.0
    max_gap: int = 2
    min_len: int = 3
    join_km: float = 20.0
    join_min: float = 20.0


@dataclass
class Track:
    tid: int
    t: list = field(default_factory=list)
    lat: list = field(default_factory=list)
    lon: list = field(default_factory=list)
    score: list = field(default_factory=list)
    misses: int = 0

    def predict(self, t: float):
        if len(self.t) < 2:
            return self.lat[-1], self.lon[-1]
        k = max(0, len(self.t) - 4)
        dt = self.t[-1] - self.t[k]
        if dt <= 0:
            return self.lat[-1], self.lon[-1]
        vy = (self.lat[-1] - self.lat[k]) / dt
        vx = (self.lon[-1] - self.lon[k]) / dt
        return self.lat[-1] + vy * (t - self.t[-1]), self.lon[-1] + vx * (t - self.t[-1])


def km(lat1, lon1, lat2, lon2):
    return np.hypot((np.asarray(lat1) - lat2) * KM_LAT, (np.asarray(lon1) - lon2) * KM_LAT * np.cos(np.radians(lat2)))


def link(det: pd.DataFrame, times: np.ndarray, p: Params) -> list[Track]:
    """det: one method's detections (t, lat, lon, score). times: all frame times (sorted)."""
    by_t = {t: g for t, g in det[det.score >= p.cont_thr].groupby("t")}
    live, done, nid = [], [], 0
    for t in times:
        g = by_t.get(t)
        cand = g[["lat", "lon", "score"]].to_numpy() if g is not None else np.zeros((0, 3))
        used = np.zeros(len(cand), bool)
        if live and len(cand):
            pred = np.array([tr.predict(t) for tr in live])
            d = km(cand[None, :, 0], cand[None, :, 1], pred[:, None, 0], pred[:, None, 1])  # (tracks, dets)
            gate = p.gate_km * (1 + np.array([tr.misses for tr in live]))[:, None] ** 0.5
            assigned = set()
            for k in np.argsort(d, axis=None):
                i, j = np.unravel_index(k, d.shape)
                if d[i, j] > gate[i, 0]:
                    continue
                if i in assigned or used[j]:
                    continue
                assigned.add(i); used[j] = True
                tr = live[i]
                tr.t.append(t); tr.lat.append(cand[j, 0]); tr.lon.append(cand[j, 1]); tr.score.append(cand[j, 2]); tr.misses = 0
            for i, tr in enumerate(live):
                if i not in assigned:
                    tr.misses += 1
        else:
            for tr in live:
                tr.misses += 1
        still = []
        for tr in live:
            (still if tr.misses <= p.max_gap else done).append(tr)
        live = still
        for j in np.where(~used)[0]:
            if cand[j, 2] >= p.start_thr:
                live.append(Track(nid, [t], [cand[j, 0]], [cand[j, 1]], [cand[j, 2]])); nid += 1
    done += live
    return [tr for tr in done if len(tr.t) >= p.min_len]


def letters(n: int) -> str:
    s = ""
    n += 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def group(tracks: list[Track], p: Params) -> pd.DataFrame:
    """Assign supercells; return long table: supercell_id, meso_id, time_utc, t, lat, lon, score, tid."""
    tracks = sorted(tracks, key=lambda tr: tr.t[0])
    sc_of = {}
    for k, tr in enumerate(tracks):
        t0, la, lo = tr.t[0], tr.lat[0], tr.lon[0]
        best, bestd = None, np.inf
        for o in tracks[:k]:
            if o.t[0] > t0 or o.t[-1] < t0 - p.join_min * 60:
                continue
            if o.t[-1] >= t0:  # alive: its position at t0
                i = int(np.searchsorted(o.t, t0))
                i = min(i, len(o.t) - 1)
                d = km(la, lo, o.lat[i], o.lon[i])
            else:
                d = km(la, lo, o.lat[-1], o.lon[-1])
            if d <= p.join_km and d < bestd:
                best, bestd = o.tid, d
        sc_of[tr.tid] = sc_of[best] if best is not None else len(set(sc_of.values()))
    # letters by supercell start, numbers by meso start
    sc_ids = sorted(set(sc_of.values()), key=lambda s: min(tr.t[0] for tr in tracks if sc_of[tr.tid] == s))
    name = {s: letters(i) for i, s in enumerate(sc_ids)}
    rows, count = [], {}
    for tr in tracks:
        s = name[sc_of[tr.tid]]
        count[s] = count.get(s, 0) + 1
        for t, la, lo, sc in zip(tr.t, tr.lat, tr.lon, tr.score):
            rows.append((s, count[s], t, la, lo, sc, tr.tid))
    df = pd.DataFrame(rows, columns=["supercell_id", "meso_id", "t", "lat", "lon", "score", "tid"])
    df["time_utc"] = pd.to_datetime(df["t"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return df


def to_tracks_csv(df: pd.DataFrame, lag_s: float = 0.0) -> str:
    """Same layout as the files in tracks/ (no tornado links; dissipation / primary_sup left blank).
    Times are MRMS frame times minus lag_s (the MRMS-vs-track time offset, common.MRMS_LAG_S)."""
    out = df.sort_values(["supercell_id", "meso_id", "t"]).assign(dissipation="", primary_sup="")
    out["time_utc"] = pd.to_datetime(out["t"] - lag_s, unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return out[["supercell_id", "meso_id", "time_utc", "lat", "lon", "dissipation", "primary_sup"]].round(
        {"lat": 5, "lon": 5}).to_csv(index=False, lineterminator="\n")
