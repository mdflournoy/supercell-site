#!/usr/bin/env python3
"""Stage 3/4: detect mid-level meso centers in every frame of one event day.

Two methods, written to the same CSV (frame, t, method, row, col, lat, lon, score):
  model     MesoNet heat-map peaks >= --min-score (keep low; the tracker applies its own thresholds)
  baseline  peaks of 3-6 km AzShear (smoothed over ~1.5 km) >= 3 with composite reflectivity >= 35 dBZ
            within 5 km; score = AzShear value. The tracker tunes the AzShear threshold on 2022.

  python ml/detect.py --tar tars/20230331.tar --model model/mesonet.pt --out detections/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DX_AZ, LAT1, LON0  # noqa: E402
from frames import Day, extract, stack_raw, static_fields, to_input  # noqa: E402
from model import MesoNet, peaks, predict_full  # noqa: E402
from add_dbz import add_columns  # noqa: E402


def rc_to_latlon(r, c):
    return LAT1 - (np.asarray(r) + 0.5) * DX_AZ, LON0 + (np.asarray(c) + 0.5) * DX_AZ


def baseline_peaks(az36: np.ndarray, ref_up: np.ndarray, min_az: float = 3.0, nms_px: int = 6):
    a = az36.astype(np.float32)
    a[az36 == -128] = 0.0
    t = torch.from_numpy(a)[None, None]
    sm = F.avg_pool2d(t, 3, stride=1, padding=1)[0, 0]
    mx = F.max_pool2d(sm[None, None], 2 * nms_px + 1, stride=1, padding=nms_px)[0, 0]
    dbz_ok = torch.from_numpy((ref_up >= (35 + 32) * 2 + 1).astype(np.float32))[None, None]
    near = F.max_pool2d(dbz_ok, 19, stride=1, padding=9)[0, 0] > 0  # within ~5 km
    sm, mx, near = sm.numpy(), mx.numpy(), near.numpy()
    rr, cc = np.where((sm >= min_az) & (sm == mx) & near)
    return np.column_stack([rr, cc, sm[rr, cc]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="detections")
    ap.add_argument("--min-score", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    day = Day(extract(Path(a.tar), Path("work-detect")))
    beam, se = static_fields()
    model = MesoNet()
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    model.eval()
    rows = []
    t0 = time.time()
    for fi in range(len(day)):
        az, ref = stack_raw(day, fi)
        prob = predict_full(model, to_input(az, ref, beam))
        name, t = day.frames.name.iloc[fi], int(day.frames.t.iloc[fi])
        for method, pk in (("model", peaks(prob, a.min_score)), ("baseline", baseline_peaks(az[0], ref[0]))):
            la, lo = rc_to_latlon(pk[:, 0], pk[:, 1])
            for (r, c, s), y, x in zip(pk, la, lo):
                rows.append((name, t, method, int(r), int(c), round(float(y), 4), round(float(x), 4), round(float(s), 4),
                             bool(se[int(r), int(c)])))
        if fi % 20 == 0:
            print(f"{day.file}: frame {fi + 1}/{len(day)} ({(time.time() - t0) / (fi + 1):.1f} s/frame)", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    det = pd.DataFrame(rows, columns=["frame", "t", "method", "row", "col", "lat", "lon", "score", "in_se"])
    add_columns(det, day).to_csv(out / f"{day.file}_detections.csv.gz", index=False)  # + dbz5, dbz10
    day.frames[["name", "t"]].to_csv(out / f"{day.file}_frames.csv", index=False)
    day.labels.to_csv(out / f"{day.file}_labels.csv.gz", index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
