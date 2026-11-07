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

from common import ML_DIR
from score_tracks import score_day
from track import Params, group, link, to_tracks_csv

VAL = ["20220322", "20220415", "20220203"]


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
        starts, conts = [0.2, 0.3, 0.4, 0.5], [0.1, 0.15, 0.2, 0.3]
    else:
        starts, conts = [6, 8, 10, 12], [3, 4, 5, 6]
    for s, c, ml, jk in itertools.product(starts, conts, [2, 3, 4, 5], [15.0, 25.0]):
        if c <= s:
            yield Params(start_thr=s, cont_thr=c, min_len=ml, join_km=jk)


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
    a = ap.parse_args()
    det_dir = Path(a.det)
    out = ML_DIR / "results"
    days = sorted(p.name[:8] for p in det_dir.glob("*_detections.csv.gz"))
    test = [d for d in days if d.startswith("2023")]
    data = {d: load(det_dir, d) for d in days}

    tuning, chosen = [], {}
    for method in ("model", "baseline"):
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
    pd.DataFrame(tuning).to_csv(out / "tuning.csv", index=False)

    scores = []
    for method, p in chosen.items():
        (out / "pred_tracks" / method).mkdir(parents=True, exist_ok=True)
        for d in days:
            det, times, lab = data[d]
            pred = run(det[det.method == method], times, p)
            (out / "pred_tracks" / method / f"{d}.csv").write_text(to_tracks_csv(pred) if len(pred) else "")
            r = score_day(pred, lab)
            scores.append({"day": d, "set": "test" if d in test else "validation", "method": method, **r})
    sc = pd.DataFrame(scores)
    sc.to_csv(out / "scores.csv", index=False)
    summ = {"settings": {m: p.__dict__ for m, p in chosen.items()},
            "test": {m: pooled(sc[(sc.set == "test") & (sc.method == m)].to_dict("records")) for m in chosen},
            "validation": {m: pooled(sc[(sc.set == "validation") & (sc.method == m)].to_dict("records")) for m in chosen}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    print(sc.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
