#!/usr/bin/env python3
"""Stages 3-4: turn per-frame detections into meso/supercell tracks, tune on the held-out 2022 days,
then score on the 2023 days.

  python ml/run_tracking.py --det DIR    (DIR holds YYYYMMDD_detections.csv.gz, _frames.csv, _labels.csv.gz)

For each method (model, baseline) the tracker settings are chosen by grid search on the held-out
2022 days (best meso CSI). Those settings are then applied unchanged to 2023. Writes ml/results/:
  pred_tracks/<method>/YYYYMMDD.csv   predicted tracks in the tracks/ layout
  scores.csv                          per day and method
  tuning.csv, summary.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ML_DIR, MRMS_LAG_S
from score_tracks import score_day
from backtrack import BackParams, RefFrames, extend
from track import Params, group, link, to_tracks_csv

VAL = ["20220322", "20220415", "20220203"]
REFL = False  # set from --refl


def load(det_dir: Path, day: str):
    det = pd.read_csv(det_dir / f"{day}_detections.csv.gz")
    frames = pd.read_csv(det_dir / f"{day}_frames.csv")
    lab = pd.read_csv(det_dir / f"{day}_labels.csv.gz", dtype={"file": str, "supercell": str, "meso": str})
    return det, np.sort(frames.t.unique()), lab


def run(det, times, p: Params):
    tracks = link(det, times, p)
    return group(tracks, p) if tracks else pd.DataFrame(columns=["supercell_id", "meso_id", "t", "lat", "lon", "score", "tid", "time_utc"])


def grid(method: str):
    if method == "model":
        starts, conts, lens = [0.15, 0.2, 0.3, 0.4, 0.5], [0.05, 0.075, 0.1, 0.15, 0.2, 0.3], [3, 6, 8, 10, 12, 15, 20]
    else:
        starts, conts, lens = [6, 8, 10, 12, 15], [3, 4, 5, 6, 8], [3, 5, 8]
    # reflectivity gates (model only; the baseline already requires >= 35 dBZ within 5 km)
    dbz = [(-99.0, -99.0)] if method != "model" or not REFL else \
        [(-99.0, t) for t in (-99.0, 50.0, 52.0, 55.0)]  # detection-level gates (30-40 dBZ) never removed anything
    for s, c, ml, jk, (dm, td) in itertools.product(starts, conts, lens, [15.0, 25.0], dbz):
        if c <= s:
            yield Params(start_thr=s, cont_thr=c, min_len=ml, join_km=jk, dbz_min=dm, trk_dbz=td)


def pooled(rows: list[dict]) -> dict:
    """Totals across days → pooled POD/FAR/CSI."""
    s = pd.DataFrame(rows).sum(numeric_only=True)
    n_ok_or_false = s.pred_mesos_ok + s.false_mesos
    return {"meso_pod": s.found_mesos / max(1, s.truth_mesos), "meso_far": s.false_mesos / max(1, n_ok_or_false),
            "meso_csi": s.found_mesos / max(1, s.truth_mesos + s.false_mesos),
            "sc_pod": s.found_supercells / max(1, s.truth_supercells),
            "sc_csi": s.found_supercells / max(1, s.truth_supercells + s.false_supercells)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--refl", action="store_true", help="also tune reflectivity gates (needs dbz5 in detections)")
    ap.add_argument("--out", default=str(ML_DIR / "results"))
    ap.add_argument("--methods", nargs="+", default=["model", "baseline"])
    ap.add_argument("--lag", type=float, default=MRMS_LAG_S, help="frame time minus track time (0 for Level II)")
    ap.add_argument("--frame-tol", type=float, default=90.0, help="reflectivity frame time tolerance (s); 160 for Level II steps")
    ap.add_argument("--frames", help="folder of extracted day folders (ml/frames.extract); enables the backward "
                                     "reflectivity-cell extension of model supercells (ml/backtrack.py)")
    a = ap.parse_args()
    global REFL
    REFL = a.refl
    det_dir = Path(a.det)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    days = sorted(p.name[:8] for p in det_dir.glob("*_detections.csv.gz"))
    test = [d for d in days if d.startswith("2023")]
    data = {d: load(det_dir, d) for d in days}

    tuning, chosen = [], {}
    for method in a.methods:
        best = None
        for p in grid(method):
            rows = []
            for d in VAL:
                det, times, lab = data[d]
                pred = run(det[det.method == method], times, p)
                rows.append(score_day(pred, lab))
            r = pooled(rows)
            tuning.append({"method": method, **p.__dict__, **r})
            if best is None or r["meso_csi"] > best[1]["meso_csi"]:
                best = (p, r)
        chosen[method] = best[0]
        print(method, best[0], {k: round(v, 3) for k, v in best[1].items()})
    back = None
    if a.frames:
        refs = {d: RefFrames(Path(a.frames) / d, a.frame_tol) for d in days}
        p = chosen["model"]
        base_pred = {d: run(data[d][0][data[d][0].method == "model"], data[d][1], p) for d in VAL}
        best = None
        for sd in (35.0, 40.0, 45.0, 50.0, 55.0):
            bp = BackParams(stop_dbz=sd)
            r = pooled([score_day(extend(base_pred[d], refs[d], bp), data[d][2]) for d in VAL])
            tuning.append({"method": "model+back", **p.__dict__, "back_stop_dbz": sd, **r})
            if best is None or r["meso_csi"] > best[1]["meso_csi"]:
                best = (bp, r)
        back = best[0]
        print("backward extension", back, {k: round(v, 3) for k, v in best[1].items()})
    pd.DataFrame(tuning).to_csv(out / "tuning.csv", index=False)

    scores = []
    for method, p in chosen.items():
        (out / "pred_tracks" / method).mkdir(parents=True, exist_ok=True)
        for d in days:
            det, times, lab = data[d]
            pred = run(det[det.method == method], times, p)
            if back and method == "model":
                pred = extend(pred, refs[d], back)
            (out / "pred_tracks" / method / f"{d}.csv").write_text(to_tracks_csv(pred, a.lag) if len(pred) else "")
            r = score_day(pred, lab)
            scores.append({"day": d, "set": "test" if d in test else "validation", "method": method, **r})
    sc = pd.DataFrame(scores)
    sc.to_csv(out / "scores.csv", index=False)
    summ = {"settings": {m: p.__dict__ for m, p in chosen.items()},
            "back": back.__dict__ if back else None,
            "test": {m: pooled(sc[(sc.set == "test") & (sc.method == m)].to_dict("records")) for m in chosen},
            "validation": {m: pooled(sc[(sc.set == "validation") & (sc.method == m)].to_dict("records")) for m in chosen}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    print(sc.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
