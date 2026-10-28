#!/usr/bin/env python3
"""MRMS AzShear along each mesocyclone track, on the MRMS timestep.

Reads the renumbered day files in tracks/ (not modified) and writes one file per day to
tracks-AzShear/YYYYMMDD.csv with a row for every mesocyclone at every MRMS time (~2 min)
within its tracked lifetime (first to last scan):

  supercell_id, meso_id, time_utc   MRMS valid time (0-2 km product)
  lat, lon                          meso position linearly interpolated to that time
  azshear_0_2km, azshear_3_6km      peak MergedAzShear (s^-1) within RADIUS_KM of the
                                    interpolated position (blank = no valid MRMS data)

The 3-6 km value uses the 3-6 km file nearest in time (within 90 s), with the position
interpolated to that file's own time. A meso tracked on a single scan gets the one MRMS
time nearest to it (within 5 min).

Each MRMS file is downloaded once: results are cached in azshear-cache/cache.csv. A run
stops starting new downloads after --max-minutes; the rest is filled on the next run.
A summary is written to azshear-cache/status.json.

Data: s3://noaa-mrms-pds/CONUS/MergedAzShear_{0-2,3-6}kmAGL_00.50/YYYYMMDD/
Usage: python scripts/azshear.py --tracks tracks --out tracks-AzShear
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import io
import json
import math
import os
import re
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BUCKET = "https://noaa-mrms-pds.s3.amazonaws.com"
PRODUCTS = {
    "azshear_0_2km": "MergedAzShear_0-2kmAGL_00.50",
    "azshear_3_6km": "MergedAzShear_3-6kmAGL_00.50",
}
RADIUS_KM = 2.0
MAX_DT = timedelta(minutes=5)
KEY_RE = re.compile(r"_(\d{8})-(\d{6})\.grib2\.gz$")
S3NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


# ----------------------------------------------------------------------------- S3 listing
_session = None


def http() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def list_times(product: str, day: str) -> dict[datetime, str]:
    """{valid_time: key} for one product and UTC day (YYYYMMDD)."""
    out, token = {}, None
    prefix = f"CONUS/{product}/{day}/"
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        for attempt in range(4):
            try:
                r = http().get(BUCKET + "/", params=params, timeout=60)
                r.raise_for_status()
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        root = ET.fromstring(r.content)
        for c in root.iter(S3NS + "Contents"):
            key = c.find(S3NS + "Key").text
            m = KEY_RE.search(key)
            if m:
                t = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                out[t] = key
        if root.findtext(S3NS + "IsTruncated") == "true":
            token = root.findtext(S3NS + "NextContinuationToken")
        else:
            return out


# ----------------------------------------------------------------------------- one MRMS file
def window_max(vals, la1, lo1, dx, dy, j_pos, lat, lon, radius_km=RADIUS_KM):
    """Max of finite grid values within radius_km (great-circle) of (lat, lon); None if none."""
    nj, ni = vals.shape
    dlat = radius_km / 111.2
    dlon = radius_km / (111.2 * max(math.cos(math.radians(lat)), 0.1))
    if j_pos:
        j0, j1 = (lat - dlat - la1) / dy, (lat + dlat - la1) / dy
    else:
        j0, j1 = (la1 - (lat + dlat)) / dy, (la1 - (lat - dlat)) / dy
    i0, i1 = (lon - dlon - lo1) / dx, (lon + dlon - lo1) / dx
    j0, j1 = max(0, math.floor(j0)), min(nj - 1, math.ceil(j1))
    i0, i1 = max(0, math.floor(i0)), min(ni - 1, math.ceil(i1))
    if j0 > j1 or i0 > i1:
        return None
    jj = np.arange(j0, j1 + 1)
    ii = np.arange(i0, i1 + 1)
    glat = (la1 + jj * dy) if j_pos else (la1 - jj * dy)
    glon = lo1 + ii * dx
    LA, LO = np.meshgrid(glat, glon, indexing="ij")
    p1, p2 = np.radians(lat), np.radians(LA)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(LO - lon) / 2) ** 2
    dist = 12742.0 * np.arcsin(np.sqrt(h))
    win = vals[j0:j1 + 1, i0:i1 + 1]
    sel = win[(dist <= radius_km) & np.isfinite(win)]
    return float(sel.max()) if sel.size else None



def peak_values(key: str, points: list[tuple[float, float]], radius_km: float = RADIUS_KM) -> dict:
    """Download one MRMS grib2.gz and return the peak value within radius_km of each point."""
    import pygrib  # imported here so worker processes load it

    t0 = time.time()
    for attempt in range(4):
        try:
            r = requests.get(f"{BUCKET}/{key}", timeout=120)
            r.raise_for_status()
            break
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))
    raw = gzip.decompress(r.content)
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as fh:
        fh.write(raw)
        path = fh.name
    try:
        grbs = pygrib.open(path)
        msg = grbs.message(1)
        vals = msg.values
        if np.ma.isMaskedArray(vals):
            vals = vals.filled(np.nan)
        vals = np.asarray(vals, dtype=np.float32)
        nj, ni = vals.shape
        la1 = msg["latitudeOfFirstGridPointInDegrees"]
        lo1 = msg["longitudeOfFirstGridPointInDegrees"]
        dx = msg["iDirectionIncrementInDegrees"]
        dy = msg["jDirectionIncrementInDegrees"]
        j_pos = bool(msg["jScansPositively"])
        grbs.close()
    finally:
        os.unlink(path)
    if lo1 > 180:
        lo1 -= 360.0
    vals[vals <= -900] = np.nan  # MRMS missing (-999) / no coverage (-99900)

    out = [window_max(vals, la1, lo1, dx, dy, j_pos, lat, lon, radius_km) for lat, lon in points]
    return {"key": key, "values": out, "bytes": len(r.content), "secs": round(time.time() - t0, 2),
            "grid": [nj, ni, la1, lo1, dx, dy, j_pos]}


# ----------------------------------------------------------------------------- track files
def read_track_file(path: Path):
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = text.split("\n")
    split = next((i for i, ln in enumerate(lines) if ln.strip().strip(",").lower() == "tornadoes"), len(lines))
    track = "\n".join(ln for ln in lines[:split] if ln.strip().strip(","))
    tail = "\n".join(lines[split:]).rstrip("\n")
    df = pd.read_csv(io.StringIO(track), dtype=str, keep_default_na=False)
    return df, tail


def write_track_file(path: Path, df: pd.DataFrame, tail: str) -> None:
    out = df.to_csv(index=False, lineterminator="\n")
    if tail:
        out += "\n" + tail + "\n"
    path.write_text(out, encoding="utf-8")


def interp_track(ts: np.ndarray, la: np.ndarray, lo: np.ndarray, t: float) -> tuple[float, float]:
    """Linear interpolation of a track at time t (epoch s); linear extrapolation at the ends."""
    if len(ts) == 1:
        return float(la[0]), float(lo[0])
    if t <= ts[0]:
        k = 0
    elif t >= ts[-1]:
        k = len(ts) - 2
    else:
        k = int(np.searchsorted(ts, t) - 1)
    span = ts[k + 1] - ts[k]
    f = 0.0 if span == 0 else (t - ts[k]) / span
    return float(la[k] + f * (la[k + 1] - la[k])), float(lo[k] + f * (lo[k + 1] - lo[k]))


def cache_key(col: str, tkey: str, lat: float, lon: float) -> str:
    return f"{col}|{tkey}|{lat:.4f}|{lon:.4f}"


# ----------------------------------------------------------------------------- main
MATCH_3_6 = timedelta(seconds=90)


def main() -> None:
    import bisect

    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="tracks")
    ap.add_argument("--out", default="tracks-AzShear")
    ap.add_argument("--cache", default="azshear-cache/cache.csv")
    ap.add_argument("--status", default="azshear-cache/status.json")
    ap.add_argument("--max-minutes", type=float, default=240)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 2)))
    a = ap.parse_args()
    started = time.time()
    status = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "errors": []}
    Path(a.cache).parent.mkdir(parents=True, exist_ok=True)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[str, str] = {}
    if Path(a.cache).exists():
        c = pd.read_csv(a.cache, dtype=str, keep_default_na=False)
        cache = dict(zip(c["key"], c["value"]))
    n_cache0 = len(cache)

    # ---- tracks: {file: {(sc, meso): (times_s, lat, lon)}}
    files = sorted(p for p in Path(a.tracks).glob("*.csv") if p.name != "id_map.csv")
    tracks: dict[Path, dict] = {}
    for p in files:
        try:
            df, _ = read_track_file(p)
            t = pd.to_datetime(df["time_utc"], utc=True, errors="coerce", format="mixed")
            df = df.assign(_t=t, _la=pd.to_numeric(df["lat"], errors="coerce"), _lo=pd.to_numeric(df["lon"], errors="coerce"))
            df = df.dropna(subset=["_t", "_la", "_lo"]).sort_values("_t")
            tr = {}
            for (sc, ms), g in df.groupby(["supercell_id", "meso_id"], sort=False):
                tr[(sc, ms)] = (np.array([x.timestamp() for x in g["_t"]]), g["_la"].to_numpy(float), g["_lo"].to_numpy(float))
            tracks[p] = tr
        except Exception as e:  # noqa: BLE001
            status["errors"].append(f"{p.name}: unreadable ({e})")

    # ---- list the MRMS files for every UTC day any track touches
    days = set()
    for tr in tracks.values():
        for ts, _, _ in tr.values():
            for x in (ts.min() - MAX_DT.total_seconds(), ts.max() + MAX_DT.total_seconds()):
                d = datetime.fromtimestamp(x, timezone.utc).date()
                while d <= datetime.fromtimestamp(ts.max() + MAX_DT.total_seconds(), timezone.utc).date():
                    days.add(d.strftime("%Y%m%d"))
                    d += timedelta(days=1)
    no_listing = set()
    avail = {}
    for col, product in PRODUCTS.items():
        merged = {}
        for d in sorted(days):
            try:
                got = list_times(product, d)
            except Exception as e:  # noqa: BLE001
                got = {}
                status["errors"].append(f"listing {product} {d}: {e}")
            if not got:
                no_listing.add(f"{product}/{d}")
            merged.update(got)
        times = sorted(merged)
        avail[col] = (times, [x.timestamp() for x in times], merged)

    def nearest(col, ts_sec, tol):
        times, secs_, merged = avail[col]
        if not times:
            return None
        k = bisect.bisect_left(secs_, ts_sec)
        cands = [j for j in (k - 1, k) if 0 <= j < len(times)]
        j = min(cands, key=lambda j: abs(secs_[j] - ts_sec))
        return times[j] if abs(secs_[j] - ts_sec) <= tol.total_seconds() else None

    # ---- every meso x MRMS time -> needed values
    work: dict[str, dict] = {}   # s3 key -> {"points": [...], "ckeys": [...]}
    series = {}                  # file -> list of row dicts (with cache keys)

    def need(col, when, lat, lon):
        ck = cache_key(col, when.strftime("%Y%m%d-%H%M%S"), lat, lon)
        if ck not in cache:
            w = work.setdefault(avail[col][2][when], {"points": [], "ckeys": []})
            if ck not in w["ckeys"]:
                w["points"].append((lat, lon))
                w["ckeys"].append(ck)
        return ck

    lo_col, mid_col = "azshear_0_2km", "azshear_3_6km"
    for p, tr in tracks.items():
        rows = []
        lo_times, lo_secs, _ = avail[lo_col]
        for (sc, ms), (ts, la, lo) in tr.items():
            if len(ts) > 1 and ts[-1] > ts[0]:
                k0, k1 = bisect.bisect_left(lo_secs, ts[0]), bisect.bisect_right(lo_secs, ts[-1])
                whens = lo_times[k0:k1]
            else:
                w1 = nearest(lo_col, float(ts[0]), MAX_DT)
                whens = [w1] if w1 else []
            for when in whens:
                plat, plon = interp_track(ts, la, lo, when.timestamp())
                row = {"supercell_id": sc, "meso_id": ms, "time_utc": when.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                       "lat": round(plat, 5), "lon": round(plon, 5), lo_col: need(lo_col, when, plat, plon), mid_col: None}
                w2 = nearest(mid_col, when.timestamp(), MATCH_3_6)
                if w2:
                    mlat, mlon = interp_track(ts, la, lo, w2.timestamp())
                    row[mid_col] = need(mid_col, w2, mlat, mlon)
                rows.append(row)
        series[p] = rows

    # ---- download + process MRMS files not yet cached, within the time budget
    jobs = sorted(work.items())
    done = failed = 0
    nbytes = 0
    secs = []
    grid = None
    deadline = started + a.max_minutes * 60
    pending = list(jobs)
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        running = {}
        while pending or running:
            while pending and len(running) < a.workers * 2 and time.time() < deadline:
                key, w = pending.pop(0)
                running[ex.submit(peak_values, key, w["points"])] = (key, w)
            if not running:
                break
            fin, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
            for f in fin:
                key, w = running.pop(f)
                try:
                    res = f.result()
                    for ck, v in zip(w["ckeys"], res["values"]):
                        cache[ck] = "" if v is None else f"{v:.6f}"
                    done += 1
                    nbytes += res["bytes"]
                    secs.append(res["secs"])
                    grid = grid or res["grid"]
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    if len(status["errors"]) < 25:
                        status["errors"].append(f"{key}: {type(e).__name__}: {e}")
            if time.time() >= deadline and pending:
                status["stopped_early"] = f"time budget ({a.max_minutes} min) reached"
        if pending:
            status["stopped_early"] = f"time budget ({a.max_minutes} min) reached"

    # ---- write tracks-AzShear/YYYYMMDD.csv
    for old in out_dir.glob("*.csv"):
        old.unlink()
    filled = blank = pending_vals = 0
    cols = ["supercell_id", "meso_id", "time_utc", "lat", "lon", lo_col, mid_col]
    for p, rows in series.items():
        out_rows = []
        for r in rows:
            r = dict(r)
            for col in (lo_col, mid_col):
                ck = r[col]
                if ck is None:
                    r[col] = ""
                    blank += 1
                elif ck in cache:
                    r[col] = cache[ck]
                    filled += cache[ck] != ""
                    blank += cache[ck] == ""
                else:
                    r[col] = ""
                    pending_vals += 1
            out_rows.append(r)
        df = pd.DataFrame(out_rows, columns=cols)
        if len(df):
            df = df.sort_values(["supercell_id", "meso_id", "time_utc"],
                                key=lambda c: c.map(lambda v: (len(v), v)) if c.name == "supercell_id" else (c.astype(int) if c.name == "meso_id" else c))
        df.to_csv(out_dir / p.name, index=False, lineterminator="\n")

    pd.DataFrame(sorted(cache.items()), columns=["key", "value"]).to_csv(a.cache, index=False, lineterminator="\n")
    status.update({
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "minutes": round((time.time() - started) / 60, 1),
        "track_files": len(tracks),
        "series_rows": sum(len(r) for r in series.values()),
        "mrms_files_needed": len(jobs), "mrms_files_done": done, "mrms_files_failed": failed,
        "mrms_files_left": len(jobs) - done - failed,
        "downloaded_mb": round(nbytes / 1e6, 1),
        "secs_per_file_median": float(np.median(secs)) if secs else None,
        "grid": grid,
        "cache_entries": len(cache), "new_cache_entries": len(cache) - n_cache0,
        "values_filled": filled, "values_blank_no_data": blank, "values_pending": pending_vals,
        "days_without_mrms": sorted(no_listing),
    })
    Path(a.status).write_text(json.dumps(status, indent=1))
    print(json.dumps(status, indent=1))


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        Path("azshear-cache").mkdir(exist_ok=True)
        Path("azshear-cache/status.json").write_text(json.dumps({"fatal": traceback.format_exc()}, indent=1))
        raise
