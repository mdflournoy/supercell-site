#!/usr/bin/env python3
"""Single-site stage 3: training patches from Level II volumes for one event day.

For each radar the analysts used that day (ml/l2/radar_assign.csv, column radar_used: the radar whose
volume started at the track time, or the nearest-radar rule on days whose track times are rounded to
the minute), every EVERY-th volume during the
times it was used (+/- 15 min) is processed (ml/l2_volume.py) and cut into P x P patches:
  * positive: one per meso that the analysts tracked with this radar at that time (random offset);
  * hard negative: up to HARD per volume centered on strong mid-level azimuthal shear (>= 0.006 s^-1
    at 3 or 4.5 km) inside the Southeast states and > 25 km from every tracked meso;
  * random: one per volume on echo >= 35 dBZ inside the Southeast states.
Labels in a patch are all tracked mesos (any radar) at the volume time. The loss mask is the
Southeast states plus 25 km around tracked mesos, within 200 km of the radar.

Inputs are stored as uint8 (each channel quantized over its clip range, l2_volume.NORM).

  python ml/l2_patches.py --day 20220405 --out patches/ --workers 4
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ML_DIR, in_southeast, read_tracks  # noqa: E402

P = 128
EVERY = 3
HARD = 2
NEAR_KM = 25.0


def qrange():
    import l2_volume as L
    lo, hi = [], []
    for ch in L.CHANNELS:
        f = ch.split("_")[0]
        if f in L.NORM:
            lo.append(L.NORM[f][2]); hi.append(L.NORM[f][3])
        elif f == "range":
            lo.append(0.0); hi.append(1.2)
        else:
            lo.append(0.0); hi.append(2.0)
    return np.array(lo, np.float32)[:, None, None], np.array(hi, np.float32)[:, None, None]


def quantize(x):
    lo, hi = qrange()
    return np.round((x.astype(np.float32) - lo) / (hi - lo) * 255).clip(0, 255).astype(np.uint8)


def dequantize(q):
    lo, hi = qrange()
    return (q.astype(np.float32) / 255 * (hi - lo) + lo).astype(np.float32)


def positions_at(tr: pd.DataFrame, t: float) -> pd.DataFrame:
    """Interpolated position of every meso whose track spans t; with the analyst radar nearest in time."""
    rows = []
    for lab, g in tr.groupby("label"):
        if g.t.iloc[0] - 60 <= t <= g.t.iloc[-1] + 60:
            la = float(np.interp(t, g.t, g.lat)); lo = float(np.interp(t, g.t, g.lon))
            k = int(np.argmin(np.abs(g.t.to_numpy() - t)))
            rows.append({"label": lab, "sc": g.supercell.iloc[0], "lat": la, "lon": lo,
                         "radar": g.radar.iloc[k] if "radar" in g else ""})
    return pd.DataFrame(rows, columns=["label", "sc", "lat", "lon", "radar"])


_SE_CACHE: dict = {}


def se_grid(lat0, lon0, site):
    if site not in _SE_CACHE:
        import l2_volume as L
        rr, cc = np.mgrid[0:L.N, 0:L.N]
        la, lo = L.rc_to_latlon(rr.ravel(), cc.ravel(), lat0, lon0)
        se = in_southeast(la, lo).reshape(L.N, L.N)
        s, _ = L.grid_polar()
        _SE_CACHE[site] = (se, s <= L.MAX_RANGE_KM)
    return _SE_CACHE[site]


def work(job):
    """Process one volume -> list of patches."""
    import requests
    import l2_volume as L
    url, t, site, tr_json, seed = job
    rng = np.random.default_rng(seed)
    tr = pd.read_json(io.StringIO(tr_json), orient="split")
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "vol"
        for attempt in range(4 if not url.startswith("/") else 0):
            try:
                r = requests.get(url, timeout=180); r.raise_for_status(); f.write_bytes(r.content); break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    return site, t, [], "download failed"
                time.sleep(5)
        if url.startswith("/"):
            f = Path(url)                                   # local file (testing)
        try:
            x, meta = L.process(f)
        except Exception as e:  # noqa: BLE001
            return site, t, [], f"process failed: {e}"
    pos = positions_at(tr, t)
    se, inrange = se_grid(meta["lat"], meta["lon"], site)
    if len(pos):
        pr, pc = L.latlon_to_rc(pos.lat.to_numpy(), pos.lon.to_numpy(), meta["lat"], meta["lon"])
    else:
        pr = pc = np.zeros(0)
    near = np.zeros_like(se)
    rad = int(NEAR_KM / L.DX_KM)
    for r0, c0 in zip(pr, pc):
        r0, c0 = int(round(r0)), int(round(c0))
        j0, j1, i0, i1 = max(0, r0 - rad), min(L.N, r0 + rad + 1), max(0, c0 - rad), min(L.N, c0 + rad + 1)
        if j0 < j1 and i0 < i1:
            jj, ii = np.mgrid[j0:j1, i0:i1]
            near[j0:j1, i0:i1] |= (jj - r0) ** 2 + (ii - c0) ** 2 <= rad * rad
    mask = (se | near) & inrange
    out = []

    def cut(rc, cc, kind):
        r0 = int(np.clip(round(rc) - P // 2, 0, L.N - P)); c0 = int(np.clip(round(cc) - P // 2, 0, L.N - P))
        m = mask[r0:r0 + P, c0:c0 + P]
        if m.mean() < 0.2:
            return
        inside = (pr >= r0) & (pr < r0 + P) & (pc >= c0) & (pc < c0 + P)
        out.append({"x": quantize(x[:, r0:r0 + P, c0:c0 + P]), "m": m.copy(), "kind": kind,
                    "pts": np.stack([pr[inside] - r0, pc[inside] - c0], 1).astype(np.float32) if inside.any()
                    else np.zeros((0, 2), np.float32)})

    # positives: mesos the analysts tracked with this radar
    for k in np.where(pos.radar.to_numpy() == site)[0] if len(pos) else []:
        if 0 <= pr[k] < L.N and 0 <= pc[k] < L.N and inrange[int(pr[k]), int(pc[k])]:
            off = rng.integers(-P // 4, P // 4 + 1, 2)
            cut(pr[k] + off[0], pc[k] + off[1], 1)
    # hard negatives: strong mid-level shear far from tracked mesos
    i3, i45 = L.CHANNELS.index("AzSh_3.0"), L.CHANNELS.index("AzSh_4.5")
    strong = (np.maximum(x[i3], x[i45]).astype(np.float32) >= 0.6) & se & inrange & ~near
    cand = np.argwhere(strong)
    for k in rng.permutation(len(cand))[:HARD]:
        cut(*cand[k], 2)
    # random echo patch
    iz = L.CHANNELS.index("Z_low")
    echo = np.argwhere((x[iz].astype(np.float32) >= 35 / 60) & se & inrange)
    if len(echo):
        cut(*echo[rng.integers(len(echo))], 3)
    return site, t, out, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--out", default="patches")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-volumes", type=int, default=400)
    a = ap.parse_args()
    from l2_assign import list_keys
    t_start = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tr = read_tracks()
    tr = tr[tr.file == a.day].sort_values(["label", "t"])
    asg = pd.read_csv(ML_DIR / "l2" / "radar_assign.csv", dtype={"file": str})
    asg = asg[asg.file == a.day]
    tr = tr.merge(asg[["label", "t", "radar_used"]].rename(columns={"radar_used": "radar"}), on=["label", "t"], how="left")
    jobs = []
    tr_json = tr[["label", "supercell", "t", "lat", "lon", "radar"]].to_json(orient="split")
    for site, g in tr.groupby("radar"):
        if not isinstance(site, str) or not site:
            continue
        days = sorted({datetime.fromtimestamp(t, timezone.utc).date() for t in (g.t.min() - 900, g.t.max() + 900)})
        keys = {}
        for dd in [days[0] + timedelta(n) for n in range((days[-1] - days[0]).days + 1)]:
            keys.update(list_keys(site, datetime.combine(dd, datetime.min.time())))
        ts = np.array(sorted(keys))
        used = np.zeros(len(ts), bool)
        for t in g.t:
            used |= np.abs(ts - t) <= 900
        sel = ts[used][::EVERY]
        jobs += [(keys[t], float(t), site, tr_json, int(t) % 100000) for t in sel]
        print(site, len(g), "track points;", used.sum(), "volumes in use ->", len(sel), flush=True)
    rng = np.random.default_rng(int(a.day))
    if len(jobs) > a.max_volumes:
        jobs = [jobs[i] for i in sorted(rng.choice(len(jobs), a.max_volumes, replace=False))]
    print(a.day, len(jobs), "volumes to process", flush=True)
    X, M, K, PTS, META = [], [], [], [], []
    fails = 0
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        for n, (site, t, patches, err) in enumerate(ex.map(work, jobs)):
            if err:
                fails += 1
                print("  ", site, t, err, flush=True)
            for p in patches:
                X.append(p["x"]); M.append(np.packbits(p["m"], axis=None)); K.append(p["kind"]); PTS.append(p["pts"])
                META.append((site, t))
            if n % 25 == 0:
                print(f"  {n}/{len(jobs)} volumes, {len(X)} patches, {(time.time() - t_start) / 60:.1f} min", flush=True)
    pts = np.concatenate([np.column_stack([np.full(len(p), i), p]) for i, p in enumerate(PTS)]) if PTS else np.zeros((0, 3))
    np.savez_compressed(out / f"l2patches_{a.day}.npz", x=np.stack(X) if X else np.zeros((0, 28, P, P), np.uint8),
                        m=np.stack(M) if M else np.zeros((0, P * P // 8), np.uint8), kind=np.array(K, np.int8),
                        pts=pts.astype(np.float32), site=np.array([s for s, _ in META]), t=np.array([t for _, t in META]),
                        day=a.day)
    k = np.array(K)
    print(f"{a.day}: {len(jobs)} volumes ({fails} failed) -> {len(X)} patches (pos {(k == 1).sum()}, hard {(k == 2).sum()}, "
          f"random {(k == 3).sum()}); {len(pts)} meso centers; {(time.time() - t_start) / 60:.1f} min")


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)


if __name__ == "__main__":
    main()
