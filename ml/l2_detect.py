#!/usr/bin/env python3
"""Single-site stage 5: run the Level II model over every radar covering the Southeast on the
held-out and test days, then merge the radars into one detection set for the tracker.

Which radar sees which place: following the analysts, each location belongs to its nearest radar,
except within CLOSE_KM of a radar, where the second-nearest radar is used (the closest one's
mid-level beams overshoot there). A radar's detections are kept only inside its own area.

  python ml/l2_detect.py plan 20230324 ...                      -> JSON job matrix (day, radars)
  python ml/l2_detect.py run --day D --radars KDGX KJAN --model mesonet_l2.pt --out det/
  python ml/l2_detect.py merge --day D --in det/ --out detdir/  -> {day}_detections/frames/labels for run_tracking

Volumes with no echo >= SKIP_DBZ (lowest sweep) inside the radar's area are skipped without
dealiasing. Merging bins the asynchronous radars into STEP_S time steps: for each step, each radar
contributes its volume nearest the step (within STEP_S / 2), so the tracker sees one frame per step.
"""
from __future__ import annotations

import argparse
import json
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

CLOSE_KM = 20.0
BUFFER_KM = 60.0       # keep detections this far outside the Southeast (tracks that leave it)
SKIP_DBZ = 40.0
STEP_S = 300.0
PEAK_THR = 0.05
RADARS_PER_JOB = 3


def sites():
    return pd.read_csv(ML_DIR / "l2" / "nexrad_sites.csv")


def km(lat0, lon0, lat1, lon1):
    p0, p1 = np.radians(lat0), np.radians(lat1)
    a = np.sin((p1 - p0) / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(np.radians(lon1 - lon0) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def responsible(lat, lon, st: pd.DataFrame) -> np.ndarray:
    """Index (into st) of the radar responsible for each point."""
    d = km(np.ravel(lat)[:, None], np.ravel(lon)[:, None], st.lat.to_numpy()[None], st.lon.to_numpy()[None])
    o = np.argsort(d, axis=1)
    first, second = o[:, 0], o[:, 1]
    close = d[np.arange(len(d)), first] < CLOSE_KM
    return np.where(close, second, first).reshape(np.shape(lat))


_REG = None


def region_mask(lat, lon):
    """Southeast states plus ~BUFFER_KM (box dilation of a 0.05-degree mask; cheap lookup)."""
    global _REG
    if _REG is None:
        la = np.arange(25.0, 44.0, 0.05); lo = np.arange(-101.0, -74.0, 0.05)
        LA, LO = np.meshgrid(la, lo, indexing="ij")
        se = in_southeast(LA, LO).astype(np.int32)
        ky = int(round(BUFFER_KM / 111.2 / 0.05)); kx = int(round(BUFFER_KM / (111.2 * np.cos(np.radians(34))) / 0.05))
        c = np.pad(se, ((ky + 1, ky), (kx + 1, kx))).cumsum(0).cumsum(1)
        box = c[2 * ky + 1:, 2 * kx + 1:] - c[:-2 * ky - 1, 2 * kx + 1:] - c[2 * ky + 1:, :-2 * kx - 1] + c[:-2 * ky - 1, :-2 * kx - 1]
        _REG = (la, lo, box > 0)
    la, lo, m = _REG
    i = np.clip(np.round((np.ravel(lat) - la[0]) / 0.05).astype(int), 0, len(la) - 1)
    j = np.clip(np.round((np.ravel(lon) - lo[0]) / 0.05).astype(int), 0, len(lo) - 1)
    return m[i, j].reshape(np.shape(lat))


def plan(days):
    st = sites()
    g = np.mgrid[28:40.6:0.05, -97.5:-77.4:0.05].reshape(2, -1).T
    keep = region_mask(g[:, 0], g[:, 1])
    idx = np.unique(responsible(g[keep, 0], g[keep, 1], st))
    radars = sorted(st.site.iloc[idx])
    jobs = [{"day": d, "radars": " ".join(radars[i:i + RADARS_PER_JOB])}
            for d in days for i in range(0, len(radars), RADARS_PER_JOB)]
    print(json.dumps(jobs))
    print(len(radars), "radars:", " ".join(radars), file=sys.stderr)


_MODEL = None
_AREA: dict = {}


def area(site, lat0, lon0):
    """Grid mask of this radar's area (responsible & region & in range)."""
    if site not in _AREA:
        import l2_volume as L
        st = sites()
        st = st[km(lat0, lon0, st.lat.to_numpy(), st.lon.to_numpy()) < 500].reset_index(drop=True)
        rr, cc = np.mgrid[0:L.N, 0:L.N]
        la, lo = L.rc_to_latlon(rr.ravel(), cc.ravel(), lat0, lon0)
        mine = st.site.to_numpy()[responsible(la, lo, st)] == site
        s, _ = L.grid_polar()
        m = mine.reshape(L.N, L.N) & (s <= L.MAX_RANGE_KM)
        reg = np.zeros(L.N * L.N, bool)
        reg[mine] = region_mask(la[mine], lo[mine])
        _AREA[site] = m & reg.reshape(L.N, L.N)
    return _AREA[site]


def quick_echo(path, site):
    """Max lowest-sweep reflectivity inside the radar's area (cheap: one sweep, one field)."""
    import pyart
    import l2_volume as L
    r = pyart.io.read_nexrad_archive(str(path), scans=[0], include_fields=["reflectivity"])
    lat0, lon0 = float(r.latitude["data"][0]), float(r.longitude["data"][0])
    z = np.ma.filled(r.fields["reflectivity"]["data"].astype(np.float32), np.nan)
    az = r.azimuth["data"]; rng = r.range["data"] / 1000
    s, gaz = L.grid_polar()
    m = area(site, lat0, lon0)
    sel = m & (s < rng[-1])
    ai = np.round((gaz[sel] - az[np.argsort(az)][0]) % 360 / np.median(np.abs(np.diff(np.sort(az))))).astype(int) % len(az)
    zs = z[np.argsort(az)]
    gi = np.clip(np.round((s[sel] - rng[0]) / (rng[1] - rng[0])).astype(int), 0, len(rng) - 1)
    v = zs[ai, gi]
    return float(np.nanmax(v)) if np.isfinite(v).any() else -99.0


def work(job):
    import requests
    import torch
    import torch.nn.functional as F
    import l2_volume as L
    from model import MesoNet, peaks
    global _MODEL
    url, t, site, model_path = job
    if _MODEL is None:
        _MODEL = MesoNet(cin=L.NCH)
        _MODEL.load_state_dict(torch.load(model_path, map_location="cpu"))
        _MODEL.eval()
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "vol"
        for attempt in range(4):
            try:
                r = requests.get(url, timeout=180); r.raise_for_status(); f.write_bytes(r.content); break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    return site, t, None, "download failed"
                time.sleep(5)
        try:
            if quick_echo(f, site) < SKIP_DBZ:
                return site, t, [], "skip"
            x, meta = L.process(f)
        except Exception as e:  # noqa: BLE001
            return site, t, None, f"process failed: {e}"
    with torch.no_grad():
        prob = torch.sigmoid(_MODEL(torch.from_numpy(x.astype(np.float32))[None]))[0].numpy()
    m = area(site, meta["lat"], meta["lon"])
    prob[~m] = 0
    pk = peaks(prob, PEAK_THR, 6)
    if not len(pk):
        return site, t, [], ""
    la, lo = L.rc_to_latlon(pk[:, 0], pk[:, 1], meta["lat"], meta["lon"])
    zl = x[L.CHANNELS.index("Z_low")].astype(np.float32) * 60
    z5 = F.max_pool2d(torch.from_numpy(zl)[None, None], 19, 1, 9)[0, 0].numpy()
    rows = [{"site": site, "vol_t": t, "row": int(r_), "col": int(c_), "lat": round(float(a), 5), "lon": round(float(b), 5),
             "score": round(float(p), 4), "dbz5": round(float(z5[int(r_), int(c_)]), 1)}
            for (r_, c_, p), a, b in zip(pk, la, lo)]
    return site, t, rows, ""


def run(a):
    from l2_assign import list_keys
    w = json.loads((ML_DIR / "l2" / "windows.json").read_text())[a.day]
    t0, t1 = w[0] - 600, w[1] + 600
    jobs = []
    for site in a.radars:
        keys = {}
        d = datetime.fromtimestamp(t0, timezone.utc).date()
        while d <= datetime.fromtimestamp(t1, timezone.utc).date():
            keys.update(list_keys(site, datetime.combine(d, datetime.min.time())))
            d += timedelta(1)
        sel = sorted(t for t in keys if t0 <= t <= t1)
        jobs += [(keys[t], float(t), site, a.model) for t in sel]
        print(site, len(sel), "volumes", flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, vols, t_start = [], [], time.time()
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        for n, (site, t, r, err) in enumerate(ex.map(work, jobs, chunksize=2)):
            vols.append({"site": site, "vol_t": t, "status": err or "ok", "n": len(r) if r else 0})
            if r:
                rows += r
            if err and err != "skip":
                print("  ", site, t, err, flush=True)
            if n % 50 == 0:
                print(f"  {n}/{len(jobs)} volumes, {len(rows)} detections, {(time.time() - t_start) / 60:.1f} min", flush=True)
    tag = f"{a.day}_{'-'.join(a.radars)}"
    pd.DataFrame(rows, columns=["site", "vol_t", "row", "col", "lat", "lon", "score", "dbz5"]).to_csv(
        out / f"l2det_{tag}.csv.gz", index=False)
    pd.DataFrame(vols).to_csv(out / f"l2vols_{tag}.csv", index=False)
    v = pd.DataFrame(vols)
    print(tag, v.status.value_counts().to_dict(), len(rows), "detections", f"{(time.time() - t_start) / 60:.1f} min")


def merge(a):
    inp, out = Path(a.inp), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    det = pd.concat([pd.read_csv(f) for f in sorted(inp.glob(f"l2det_{a.day}_*.csv.gz"))], ignore_index=True)
    vols = pd.concat([pd.read_csv(f) for f in sorted(inp.glob(f"l2vols_{a.day}_*.csv"))], ignore_index=True)
    vols = vols[vols.status.isin(["ok", "skip"])]
    w = json.loads((ML_DIR / "l2" / "windows.json").read_text())[a.day]
    steps = np.arange(w[0], w[1] + 1, STEP_S)
    rows = []
    for site, v in vols.groupby("site"):
        vt = np.sort(v.vol_t.unique())
        k = np.clip(np.searchsorted(vt, steps), 1, len(vt) - 1) if len(vt) > 1 else np.zeros(len(steps), int)
        cand = np.where(np.abs(vt[k - 1] - steps) < np.abs(vt[k] - steps), vt[k - 1], vt[k]) if len(vt) > 1 else vt[k]
        ok = np.abs(cand - steps) <= STEP_S / 2
        use = dict(zip(cand[ok], steps[ok]))            # volume -> step (a volume serves at most one step)
        d = det[(det.site == site) & det.vol_t.isin(list(use))]
        if len(d):
            rows.append(d.assign(t=d.vol_t.map(use)))
    det = pd.concat(rows, ignore_index=True) if rows else det.assign(t=[])
    det["frame"] = pd.to_datetime(det.t, unit="s").dt.strftime("%Y%m%d-%H%M%S")
    det["method"] = "model"
    det["in_se"] = in_southeast(det.lat.to_numpy(), det.lon.to_numpy())
    det["dbz10"] = det.dbz5
    det[["frame", "t", "method", "row", "col", "lat", "lon", "score", "in_se", "site", "vol_t", "dbz5", "dbz10"]].to_csv(
        out / f"{a.day}_detections.csv.gz", index=False)
    pd.DataFrame({"name": pd.to_datetime(steps, unit="s").strftime("%Y%m%d-%H%M%S"), "t": steps}).to_csv(
        out / f"{a.day}_frames.csv", index=False)
    # labels: hand-made positions interpolated to each step (track times are volume start times)
    tr = read_tracks()
    tr = tr[tr.file == a.day]
    lab = []
    for (sc, ms), g in tr.groupby(["supercell", "meso"]):
        g = g.sort_values("t")
        ts = steps[(steps >= g.t.iloc[0]) & (steps <= g.t.iloc[-1])]
        if not len(ts):
            continue
        la = np.interp(ts, g.t, g.lat); lo = np.interp(ts, g.t, g.lon)
        lab.append(pd.DataFrame({"frame": pd.to_datetime(ts, unit="s").strftime("%Y%m%d-%H%M%S"), "t": ts, "file": a.day,
                                 "supercell": sc, "meso": ms, "label": g.label.iloc[0], "lat": la, "lon": lo,
                                 "tornadic": g.tornadic.iloc[0], "sc_tornadic": g.sc_tornadic.iloc[0],
                                 "row": np.nan, "col": np.nan}))
    lab = pd.concat(lab, ignore_index=True)
    lab["in_se"] = in_southeast(lab.lat.to_numpy(), lab.lon.to_numpy())
    lab.to_csv(out / f"{a.day}_labels.csv.gz", index=False)
    print(a.day, len(det), "detections in", len(steps), "steps;", len(lab), "label points")


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    import tensorflow as tf
    import torch
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("days", nargs="+")
    r = sub.add_parser("run"); r.add_argument("--day", required=True); r.add_argument("--radars", nargs="+", required=True)
    r.add_argument("--model", required=True); r.add_argument("--out", default="l2det"); r.add_argument("--workers", type=int, default=4)
    m = sub.add_parser("merge"); m.add_argument("--day", required=True); m.add_argument("--in", dest="inp", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    {"plan": lambda: plan(a.days), "run": lambda: run(a), "merge": lambda: merge(a)}[a.cmd]()


if __name__ == "__main__":
    main()
