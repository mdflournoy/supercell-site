#!/usr/bin/env python3
"""Stage 1: build the MRMS input frames and labels for one track file (one event day).

For the file's analysis window (first tracked scan - 1 h to last scan + 1 h), every other MRMS
0-2 km AzShear time (~4 min) becomes a frame. For each frame, the 3-6 km AzShear and the composite
reflectivity files nearest in time (within 90 s) are fetched from s3://noaa-mrms-pds and cropped to
the Southeast box (see common.py).

Output folder (--out), one sub-folder per track file:
  frames/YYYYMMDD-HHMMSS.npz   az02, az36: int8 AzShear (10^-3 s^-1; -128 = no data)
                               ref: uint8 composite reflectivity, (dBZ + 32) * 2 + 1 (0 = no data)
  frames.csv                   frame time and the MRMS keys/times used
  labels.csv                   every tracked meso active at each frame (all track files), with
                               position, grid row/col, Southeast flag and tornadic flags
  offsets.csv                  for each label: where the 3-6 km AzShear maximum within 6 km lies
                               relative to the tracked position (checks time/position alignment)
  status.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import math
import os
import re
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DX_AZ, DX_REF, LAT1, LON0, analysis_windows, box_shape, in_southeast,  # noqa: E402
                    latlon_to_rc, meso_positions_at, read_tracks)

BUCKET = "https://noaa-mrms-pds.s3.amazonaws.com"
AZ02 = "MergedAzShear_0-2kmAGL_00.50"
AZ36 = "MergedAzShear_3-6kmAGL_00.50"
REF_CANDIDATES = ["MergedReflectivityQCComposite_00.50", "MergedReflectivityQComposite_00.50"]
KEY_RE = re.compile(r"_(\d{8})-(\d{6})\.grib2\.gz$")
S3NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
MATCH = 90  # s
FRAME_STEP = 240  # s


def get(url, **kw):
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=120, **kw)
            r.raise_for_status()
            return r
        except Exception:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))


def list_times(product: str, day: str) -> dict[float, str]:
    out, token = {}, None
    while True:
        params = {"list-type": "2", "prefix": f"CONUS/{product}/{day}/", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        root = ET.fromstring(get(BUCKET + "/", params=params).content)
        for c in root.iter(S3NS + "Contents"):
            key = c.find(S3NS + "Key").text
            m = KEY_RE.search(key)
            if m:
                t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                out[t.timestamp()] = key
        if root.findtext(S3NS + "IsTruncated") == "true":
            token = root.findtext(S3NS + "NextContinuationToken")
        else:
            return out


def list_window(product: str, t0: float, t1: float) -> dict[float, str]:
    d = datetime.fromtimestamp(t0, timezone.utc).date()
    out = {}
    while d <= datetime.fromtimestamp(t1, timezone.utc).date():
        out.update(list_times(product, d.strftime("%Y%m%d")))
        d += timedelta(days=1)
    return {t: k for t, k in out.items() if t0 - MATCH <= t <= t1 + MATCH}


def nearest(times: np.ndarray, t: float):
    if not len(times):
        return None
    i = int(np.abs(times - t).argmin())
    return times[i] if abs(times[i] - t) <= MATCH else None


def fetch_box(key: str, dx: float) -> np.ndarray:
    """Download one MRMS grib2.gz and return the Southeast box (float32, NaN = missing)."""
    import pygrib

    raw = gzip.decompress(get(f"{BUCKET}/{key}").content)
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as fh:
        fh.write(raw)
        path = fh.name
    try:
        grbs = pygrib.open(path)
        msg = grbs.message(1)
        la1 = msg["latitudeOfFirstGridPointInDegrees"]
        lo1 = msg["longitudeOfFirstGridPointInDegrees"]
        if lo1 > 180:
            lo1 -= 360.0
        gdx = msg["iDirectionIncrementInDegrees"]
        assert abs(gdx - dx) < 1e-6, (key, gdx, dx)
        assert not msg["jScansPositively"], key
        vals = msg.values
        grbs.close()
    finally:
        os.unlink(path)
    if np.ma.isMaskedArray(vals):
        vals = vals.filled(np.nan)
    j0 = (la1 - (LAT1 - dx / 2)) / dx
    i0 = (LON0 + dx / 2 - lo1) / dx
    assert abs(j0 - round(j0)) < 1e-3 and abs(i0 - round(i0)) < 1e-3, (key, j0, i0)
    ny, nx = box_shape(dx)
    j0, i0 = int(round(j0)), int(round(i0))
    return np.asarray(vals[j0:j0 + ny, i0:i0 + nx], dtype=np.float32)


def q_az(a: np.ndarray):
    a = a.copy()
    a[(a == 0) | (a <= -900)] = np.nan  # MRMS: missing / no coverage / range-folded are 0
    frac = np.nanmax(np.abs(a - np.round(a))) if np.isfinite(a).any() else 0.0
    out = np.where(np.isfinite(a), np.clip(np.round(a), -127, 127), -128).astype(np.int8)
    return out, float(frac)


def q_ref(a: np.ndarray) -> np.ndarray:
    ok = np.isfinite(a) & (a > -90)
    return np.where(ok, np.clip(np.round((a + 32) * 2) + 1, 1, 255), 0).astype(np.uint8)


def build_frame(fr: dict, out_dir: Path) -> dict:
    t0 = time.time()
    az02, f1 = q_az(fetch_box(fr["key_az02"], DX_AZ))
    az36, f2 = q_az(fetch_box(fr["key_az36"], DX_AZ))
    ref = q_ref(fetch_box(fr["key_ref"], DX_REF))
    name = datetime.fromtimestamp(fr["t"], timezone.utc).strftime("%Y%m%d-%H%M%S")
    np.savez_compressed(out_dir / "frames" / f"{name}.npz", az02=az02, az36=az36, ref=ref)
    return {"t": fr["t"], "name": name, "secs": round(time.time() - t0, 1), "nonint": max(f1, f2),
            "az36_valid": float((az36 != -128).mean()), "ref_valid": float((ref > 0).mean())}


def offsets(labels: pd.DataFrame, frames_dir: Path, radius_km: float = 6.0) -> pd.DataFrame:
    """Location of the 3-6 km AzShear max within radius_km of each label, relative to the label."""
    rows = []
    rad = int(math.ceil(radius_km / (DX_AZ * 111.2))) + 1
    for name, g in labels.groupby("frame"):
        f = frames_dir / f"{name}.npz"
        if not f.exists():
            continue
        az = np.load(f)["az36"].astype(np.float32)
        az[az == -128] = np.nan
        for _, r in g.iterrows():
            rr, cc = int(round(r.row)), int(round(r.col))
            j0, j1 = max(0, rr - rad), min(az.shape[0], rr + rad + 1)
            i0, i1 = max(0, cc - rad), min(az.shape[1], cc + rad + 1)
            win = az[j0:j1, i0:i1]
            if not np.isfinite(win).any():
                rows.append((name, r.file, r.label, np.nan, np.nan, np.nan))
                continue
            jj, ii = np.mgrid[j0:j1, i0:i1]
            dy = -(jj - r.row) * DX_AZ * 111.2
            dxk = (ii - r.col) * DX_AZ * 111.2 * math.cos(math.radians(r.lat))
            w = np.where(np.hypot(dy, dxk) <= radius_km, win, np.nan)
            if not np.isfinite(w).any():
                rows.append((name, r.file, r.label, np.nan, np.nan, np.nan))
                continue
            k = np.nanargmax(w)
            rows.append((name, r.file, r.label, float(dxk.flat[k]), float(dy.flat[k]), float(w.flat[k])))
    return pd.DataFrame(rows, columns=["frame", "file", "label", "dx_east_km", "dy_north_km", "az36_max"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="track file stem, e.g. 20220405")
    ap.add_argument("--out", default="ml-out")
    ap.add_argument("--workers", type=int, default=max(2, os.cpu_count() or 2))
    ap.add_argument("--max-frames", type=int, default=0, help="testing: stop after this many frames")
    a = ap.parse_args()
    started = time.time()
    tr = read_tracks()
    w0, w1 = analysis_windows(tr)[a.file]
    out_dir = Path(a.out) / a.file
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    status = {"file": a.file, "window": [datetime.fromtimestamp(x, timezone.utc).isoformat() for x in (w0, w1)]}

    lists = {p: list_window(p, w0, w1) for p in (AZ02, AZ36)}
    ref_product = None
    for p in REF_CANDIDATES:
        lst = list_window(p, w0, w1)
        if lst:
            ref_product, lists["ref"] = p, lst
            break
    if ref_product is None:
        raise SystemExit("no reflectivity product found")
    status["ref_product"] = ref_product
    t02 = np.array(sorted(lists[AZ02]))
    t36 = np.array(sorted(lists[AZ36]))
    tre = np.array(sorted(lists["ref"]))

    # frames: the 0-2 km time nearest each 4-min step of the window
    frames, used = [], set()
    for t in np.arange(math.ceil(w0 / FRAME_STEP) * FRAME_STEP, w1 + 1, FRAME_STEP):
        ta = nearest(t02, t)
        if ta is None or ta in used:
            continue
        tb, tc = nearest(t36, ta), nearest(tre, ta)
        if tb is None or tc is None:
            continue
        used.add(ta)
        frames.append({"t": ta, "key_az02": lists[AZ02][ta], "key_az36": lists[AZ36][tb], "key_ref": lists["ref"][tc],
                       "dt_az36": tb - ta, "dt_ref": tc - ta})
    if a.max_frames:
        frames = frames[:a.max_frames]
    status["frames_planned"] = len(frames)
    status["window_steps"] = int((w1 - w0) // FRAME_STEP)

    done, errors = [], []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(build_frame, fr, out_dir): fr for fr in frames}
        for f in cf.as_completed(futs):
            try:
                done.append(f.result())
            except Exception as e:  # noqa: BLE001
                errors.append(f"{futs[f]['key_az02']}: {e!r}")
    meta = pd.DataFrame(frames).merge(pd.DataFrame(done), on="t", how="inner").sort_values("t")
    meta["time_utc"] = pd.to_datetime(meta["t"], unit="s", utc=True)
    meta.to_csv(out_dir / "frames.csv", index=False)

    # labels: every tracked meso (from any file) active at each frame
    lab = []
    for _, fr in meta.iterrows():
        p = meso_positions_at(tr, fr["t"])
        if len(p):
            p.insert(0, "frame", fr["name"])
            p.insert(1, "t", fr["t"])
            lab.append(p)
    labels = pd.concat(lab, ignore_index=True) if lab else pd.DataFrame(
        columns=["frame", "t", "file", "supercell", "meso", "label", "lat", "lon", "tornadic", "sc_tornadic"])
    if len(labels):
        r, c = latlon_to_rc(labels["lat"], labels["lon"])
        labels["row"], labels["col"] = np.round(r, 2), np.round(c, 2)
        labels["in_se"] = in_southeast(labels["lat"].to_numpy(), labels["lon"].to_numpy())
    labels.to_csv(out_dir / "labels.csv", index=False)
    off = offsets(labels, out_dir / "frames") if len(labels) else pd.DataFrame()
    off.to_csv(out_dir / "offsets.csv", index=False)

    sizes = [f.stat().st_size for f in (out_dir / "frames").glob("*.npz")]
    status.update({
        "frames_done": len(meta), "errors": errors[:20], "n_errors": len(errors), "labels": len(labels),
        "mesos": int(labels.groupby(["file", "label"]).ngroups) if len(labels) else 0,
        "mb_per_frame": round(float(np.mean(sizes)) / 1e6, 2) if sizes else None,
        "total_mb": round(sum(sizes) / 1e6, 1), "max_nonint": float(meta["nonint"].max()) if len(meta) else None,
        "median_secs_per_frame": float(meta["secs"].median()) if len(meta) else None,
        "max_abs_dt_az36": float(meta["dt_az36"].abs().max()) if len(meta) else None,
        "max_abs_dt_ref": float(meta["dt_ref"].abs().max()) if len(meta) else None,
        "minutes": round((time.time() - started) / 60, 1),
    })
    (out_dir / "status.json").write_text(json.dumps(status, indent=1))
    print(json.dumps(status, indent=1))


if __name__ == "__main__":
    main()
