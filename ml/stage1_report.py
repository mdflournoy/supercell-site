#!/usr/bin/env python3
"""Stage 1 summary from the per-day CSVs of the ml-frames-v1 release (frames, labels, offsets, status).

Usage: python ml/stage1_report.py --assets DIR   (DIR holds YYYYMMDD_{frames,labels,offsets}.csv and _status.json)
Writes ml/diagnostics/stage1_days.csv, offsets_summary.json and offsets.png.

Offsets: for each tracked meso at each frame, where the 3-6 km AzShear maximum within 6 km sits relative
to the tracked position, split into along-motion and across-motion parts. A systematic along-motion
offset means MRMS and the Level II tracks are shifted in time; offset / speed estimates that lag.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ML_DIR, read_tracks


def motion(tr: pd.DataFrame, file: str, label: str, t: float, half: float = 300.0):
    g = tr[(tr.file == file) & (tr.label == label)]
    ts = g.t.to_numpy()
    if len(ts) < 2:
        return np.nan, np.nan
    a, b = max(ts[0], t - half), min(ts[-1], t + half)
    if b - a < 120:
        return np.nan, np.nan
    la = np.interp([a, b], ts, g.lat.to_numpy())
    lo = np.interp([a, b], ts, g.lon.to_numpy())
    u = (lo[1] - lo[0]) * 111.2 * np.cos(np.radians(la.mean())) * 1000 / (b - a)
    v = (la[1] - la[0]) * 111.2 * 1000 / (b - a)
    return u, v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    a = ap.parse_args()
    d = Path(a.assets)
    out = ML_DIR / "diagnostics"
    out.mkdir(exist_ok=True)
    tr = read_tracks()

    days = []
    for s in sorted(d.glob("*_status.json")):
        st = json.loads(s.read_text())
        f = st["file"]
        lab = pd.read_csv(d / f"{f}_labels.csv") if (d / f"{f}_labels.csv").exists() else pd.DataFrame()
        own = lab[lab.file.astype(str) == f] if len(lab) else lab
        days.append({"file": f, "year": f[:4], "frames": st["frames_done"], "planned": st["frames_planned"],
                     "window_steps": st["window_steps"], "errors": st["n_errors"], "total_mb": st["total_mb"],
                     "labels": len(lab), "mesos_own": own.groupby("label").ngroups if len(own) else 0,
                     "labels_outside_se": int((~lab.in_se.astype(bool)).sum()) if len(lab) else 0,
                     "max_dt_ref_s": st["max_abs_dt_ref"], "minutes": st["minutes"]})
    days = pd.DataFrame(days)
    days.to_csv(out / "stage1_days.csv", index=False)

    offs = []
    for f in sorted(d.glob("*_offsets.csv")):
        try:
            o = pd.read_csv(f, dtype={"file": str})
        except pd.errors.EmptyDataError:
            continue
        if not len(o):
            continue
        lab = pd.read_csv(d / f.name.replace("_offsets", "_labels"), dtype={"file": str})
        o = o.merge(lab[["frame", "file", "label", "t", "lat", "lon", "tornadic"]], on=["frame", "file", "label"], how="left")
        offs.append(o)
    o = pd.concat(offs, ignore_index=True).drop_duplicates(["frame", "file", "label"])
    uv = [motion(tr, r.file, r.label, r.t) for r in o.itertuples()]
    o["u"], o["v"] = [x[0] for x in uv], [x[1] for x in uv]
    spd = np.hypot(o.u, o.v)
    ok = spd > 3
    o["along_km"] = np.where(ok, (o.dx_east_km * o.u + o.dy_north_km * o.v) / spd, np.nan)
    o["across_km"] = np.where(ok, (-o.dx_east_km * o.v + o.dy_north_km * o.u) / spd, np.nan)  # + = left of motion
    o["lag_min"] = o["along_km"] * 1000 / spd / 60  # + = AzShear max ahead of the tracked position
    o["dist_km"] = np.hypot(o.dx_east_km, o.dy_north_km)
    o["year"] = o.file.str[:4]
    o.to_csv(out / "offsets_all.csv.gz", index=False)

    v = o.dropna(subset=["along_km"])
    summ = {
        "labels": int(len(o)), "with_az_value": int(o.az36_max.notna().sum()),
        "dist_km_median": round(float(o.dist_km.median()), 2),
        "within_2km": round(float((o.dist_km <= 2).mean()), 3), "within_4km": round(float((o.dist_km <= 4).mean()), 3),
        "along_km_median": round(float(v.along_km.median()), 2), "across_km_median": round(float(v.across_km.median()), 2),
        "along_km_mean": round(float(v.along_km.mean()), 2), "storm_speed_median": round(float(spd[ok].median()), 1),
        "lag_min_median": round(float(v.lag_min.median()), 2),
        "by_year": {y: {"along_km_median": round(float(g.along_km.median()), 2), "n": int(len(g))} for y, g in v.groupby("year")},
    }
    (out / "offsets_summary.json").write_text(json.dumps(summ, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    a0 = ax[0]
    h = a0.hist2d(v.across_km, v.along_km, bins=np.arange(-6, 6.01, 0.5), cmap="Greys")
    a0.axhline(0, color="#e34948", lw=0.8); a0.axvline(0, color="#e34948", lw=0.8)
    a0.set_aspect("equal"); a0.set_xlabel("Across motion (km, + = left)"); a0.set_ylabel("Along motion (km, + = ahead)")
    a0.set_title("Where the 3–6 km AzShear max sits\nrelative to the tracked position")
    a1 = ax[1]
    a1.hist(o.dist_km.dropna(), bins=np.arange(0, 6.01, 0.25), color="0.4")
    a1.set_xlabel("Distance from tracked position (km)"); a1.set_ylabel("Meso-frames")
    a1.set_title(f"Median {summ['dist_km_median']} km · {summ['within_2km']:.0%} within 2 km")
    a2 = ax[2]
    a2.hist(v.lag_min.clip(-10, 10), bins=np.arange(-10, 10.01, 0.5), color="#2a78d6")
    a2.axvline(0, color="k", lw=0.8); a2.axvline(summ["lag_min_median"], color="#e34948", lw=1.5)
    a2.set_xlabel("Along-motion offset as time (min, + = MRMS ahead)"); a2.set_ylabel("Meso-frames")
    a2.set_title(f"Median {summ['lag_min_median']} min")
    fig.savefig(out / "offsets.png", dpi=130)
    print(days.to_string(index=False))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
