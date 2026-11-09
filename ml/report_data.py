#!/usr/bin/env python3
"""Stage 5: data for the comparison report (predicted vs. hand-made tracks on each day).

  python ml/report_data.py --det DIR --results ml/results/v2 [--label "MesoNet v2"]

Re-runs tracking with the settings chosen in RESULTS/summary.json, scores each day with per-meso
outcomes, and writes RESULTS/report.json: hand-made tracks (track times), predicted tracks
(MRMS frame time - MRMS_LAG_S), outcomes, per-day and pooled scores, and Southeast outlines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ML_DIR, MRMS_LAG_S, read_tracks
from run_tracking import load, pooled, run
from score_tracks import score_day
from backtrack import BackParams, RefFrames, extend
from track import Params


def clean(v):
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--frames", help="extracted day folders; needed when summary.json has a backward extension")
    a = ap.parse_args()
    det_dir, res_dir = Path(a.det), Path(a.results)
    summ = json.loads((res_dir / "summary.json").read_text())
    settings = {m: Params(**v) for m, v in summ["settings"].items()}
    back = BackParams(**summ["back"]) if summ.get("back") else None
    if back and not a.frames:
        raise SystemExit("summary.json uses the backward extension: pass --frames")
    tracks = read_tracks()
    days = sorted(p.name[:8] for p in det_dir.glob("*_detections.csv.gz"))
    out_days = []
    for d in days:
        det, times, lab = load(det_dir, d)
        tr = tracks[tracks.file == d]
        entry = {"day": d, "set": "test" if d.startswith("2023") else "validation", "pred": {}, "scores": {}}
        found_any = {}
        for m, p in settings.items():
            pred = run(det[det.method == m], times, p)
            if back and m == "model":
                pred = extend(pred, RefFrames(Path(a.frames) / d), back)
            r, found, pst, main_of = score_day(pred, lab, status=True)
            entry["scores"][m] = {k: clean(v) for k, v in r.items()}
            found_any[m] = found
            pm = []
            if len(pred):
                pred = pred.assign(pkey=pred.supercell_id + pred.meso_id.astype(str))
                for k, g in pred.groupby("pkey", sort=False):
                    g = g.sort_values("t")
                    pm.append({"id": k, "sc": g.supercell_id.iloc[0], "status": pst.get(k, "outside"),
                               "score": round(float(g.score.max()), 3),
                               "nback": int((g.stage == "back").sum()) if "stage" in g else 0,
                               "pts": [[int(t - MRMS_LAG_S), round(la, 4), round(lo, 4)] for t, la, lo in zip(g.t, g.lat, g.lon)]})
            entry["pred"][m] = pm
        tm = []
        for (sc, ms), g in tr.groupby(["supercell", "meso"], sort=False):
            g = g.sort_values("t")
            key = f"{d}:{g.label.iloc[0]}"
            tm.append({"id": g.label.iloc[0], "sc": sc, "tornadic": bool(g.tornadic.any()),
                       "found": {m: bool(f.get(key, False)) for m, f in found_any.items()},
                       "scored": key in found_any.get("model", {}),
                       "pts": [[int(t), round(la, 4), round(lo, 4)] for t, la, lo in zip(g.t, g.lat, g.lon)]})
        entry["truth"] = tm
        out_days.append(entry)
        print(d, {m: (entry["scores"][m]["found_mesos"], entry["scores"][m]["truth_mesos"], entry["scores"][m]["false_mesos"]) for m in settings})
    pooled_out = {}
    for s in ("validation", "test"):
        pooled_out[s] = {m: {k: round(v, 4) for k, v in pooled([e["scores"][m] for e in out_days if e["set"] == s]).items()}
                         for m in settings}
    se_names = {f["properties"].get("name") for f in json.loads((ML_DIR / "static" / "se_states.geojson").read_text())["features"]}
    gj = json.loads((ML_DIR / "static" / "us_states.geojson").read_text())
    states = []
    for f in gj["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        states.append({"name": f["properties"].get("name", ""), "se": f["properties"].get("name") in se_names,
                       "rings": [[[round(x, 3), round(y, 3)] for x, y in ring] for poly in polys for ring in poly]})
    rep = {"label": a.label, "lag_s": MRMS_LAG_S, "settings": summ["settings"], "back": summ.get("back"), "pooled": pooled_out,
           "days": out_days, "states": states}
    (res_dir / "report.json").write_text(json.dumps(rep, separators=(",", ":")))
    print(json.dumps(pooled_out, indent=1))


if __name__ == "__main__":
    main()
