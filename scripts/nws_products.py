#!/usr/bin/env python3
"""NWS products (tornado warnings, severe thunderstorm warnings, special weather statements) near the tracks.

Two parts:

1. fetch (nightly, needs network): for every event file in tracks/, download from the Iowa Environmental Mesonet
   archive the storm-based TO.W / SV.W polygons as issued (geojson/sbw.geojson, status NEW) and the SPS polygons
   (cgi-bin/request/gis/sps.py, a zipped shapefile) issued during the event's tracks (+/- a margin) inside a box
   around them, and cache them in nws-cache/<event>.json. An event is only downloaded again when its tracks
   extend beyond the cached time span or box, or the cache format changes.

2. associate (scripts/build_data.py): a product belongs to a mesocyclone when the meso's track, sampled every
   minute from ASSOC_BEFORE_S before issuance to ASSOC_AFTER_S after it (and not after expiration), comes inside
   the polygon or within BUFFER_KM of its edge. Only products associated with at least one meso are shown.

  python scripts/nws_products.py --tracks tracks --cache nws-cache [--max-minutes 30]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

VERSION = 1
IEM = "https://mesonet.agron.iastate.edu"
MARGIN_S = 3 * 3600        # products issued up to 3 h before the first / 1 h after the last track point
MARGIN_AFTER_S = 3600
BOX_DEG = 1.5              # box around the tracks
ASSOC_BEFORE_S = 600       # association window: issuance - 10 min ...
ASSOC_AFTER_S = 1800       # ... to issuance + 30 min (and no later than expiration)
BUFFER_KM = 5.0
KINDS = {("TO", "W"): "TO", ("SV", "W"): "SV"}


def get(url, params=None, timeout=180):
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": "supercell-site nightly build"})
            r.raise_for_status()
            return r
        except Exception:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def epoch(v) -> int | None:
    if v is None or v == "":
        return None
    v = str(v)
    if len(v) == 12 and v.isdigit():           # YYYYMMDDHHMI (SPS shapefile)
        return int(datetime.strptime(v, "%Y%m%d%H%M").replace(tzinfo=timezone.utc).timestamp())
    ts = pd.Timestamp(v)
    return int((ts.tz_localize("UTC") if ts.tzinfo is None else ts).timestamp())


def rings(geom: dict) -> list[list[list[float]]]:
    """Exterior rings of a GeoJSON (Multi)Polygon as [[lat, lon], ...] (warnings have no holes)."""
    if not geom:
        return []
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]] if geom["type"] == "Polygon" else []
    return [[[round(y, 4), round(x, 4)] for x, y, *_ in p[0]] for p in polys if p and len(p[0]) >= 4]


def in_box(rr, box) -> bool:
    la0, la1, lo0, lo1 = box
    return any(la0 <= la <= la1 and lo0 <= lo <= lo1 for r in rr for la, lo in r)


# ---------------------------------------------------------------- fetch
def fetch_sbw(sts: float, ets: float, box) -> list[dict]:
    d = get(f"{IEM}/geojson/sbw.geojson", {"sts": iso(sts), "ets": iso(ets)}).json()
    out = []
    for f in d.get("features", []):
        p = f["properties"]
        kind = KINDS.get((p.get("phenomena"), p.get("significance")))
        if not kind:
            continue
        rr = rings(f.get("geometry"))
        if not rr or not in_box(rr, box):
            continue
        out.append({
            "k": kind, "wfo": p.get("wfo"), "n": p.get("eventid"), "iss": epoch(p.get("issue") or p.get("polygon_begin")),
            "exp": epoch(p.get("expire_utc") or p.get("expire")),
            "tags": {k: p.get(k) for k in ("tornadotag", "damagetag", "windtag", "hailtag", "is_pds", "is_emergency",
                                          "max_windtag", "max_hailtag", "max_is_pds", "max_is_emergency") if p.get(k) not in (None, "", False)},
            "url": p.get("href"), "pid": p.get("product_id"), "poly": rr,
        })
    return out


def fetch_sps(sts: float, ets: float, box) -> list[dict]:
    import shapefile  # pyshp
    r = get(f"{IEM}/cgi-bin/request/gis/sps.py", {"sts": iso(sts), "ets": iso(ets)})
    if not r.content.startswith(b"PK"):                # "ERROR: no results found for your query"
        return []
    z = zipfile.ZipFile(io.BytesIO(r.content))
    stem = next(n[:-4] for n in z.namelist() if n.endswith(".shp"))
    sf = shapefile.Reader(shp=io.BytesIO(z.read(stem + ".shp")), shx=io.BytesIO(z.read(stem + ".shx")),
                          dbf=io.BytesIO(z.read(stem + ".dbf")))
    out = []
    for sr in sf.iterShapeRecords():
        p = sr.record.as_dict()
        rr = rings(sr.shape.__geo_interface__)
        if not rr or not in_box(rr, box):
            continue
        tags = {"landspout": p.get("LNDSPOUT"), "waterspout": p.get("WTRSPOUT"), "hail": p.get("MAX_HAIL"), "wind": p.get("MAX_WIND")}
        out.append({
            "k": "SPS", "wfo": p.get("WFO"), "n": None, "iss": epoch(p.get("ISSUE")), "exp": epoch(p.get("EXPIRE")),
            "tags": {k: v for k, v in tags.items() if v not in (None, "", "None")},
            "url": f"{IEM}/p.php?pid={p.get('PROD_ID')}", "pid": p.get("PROD_ID"), "poly": rr,
        })
    return out


def read_points(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = text.split("\n")
    split = next((i for i, ln in enumerate(lines) if ln.strip().strip(",").lower() == "tornadoes"), len(lines))
    df = pd.read_csv(io.StringIO("\n".join(ln for ln in lines[:split] if ln.strip().strip(","))), dtype=str, keep_default_na=False)
    t = pd.to_datetime(df["time_utc"], utc=True, errors="coerce", format="mixed")
    df["t"] = (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    df["la"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lo"] = pd.to_numeric(df["lon"], errors="coerce")
    return df.dropna(subset=["t", "la", "lo"])


def fetch_main(a) -> None:
    t_start = time.time()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(a.tracks).glob("*.csv") if p.stem[:8].isdigit())
    n_new = n_fail = 0
    for f in files:
        if (time.time() - t_start) / 60 > a.max_minutes:
            print("time budget reached; the rest resumes next run")
            break
        try:
            df = read_points(f)
        except Exception as e:  # noqa: BLE001
            print(f"{f.name}: unreadable ({e})")
            continue
        if df.empty:
            continue
        sts, ets = df.t.min() - MARGIN_S, df.t.max() + MARGIN_AFTER_S
        box = [round(df.la.min() - BOX_DEG, 2), round(df.la.max() + BOX_DEG, 2), round(df.lo.min() - BOX_DEG, 2), round(df.lo.max() + BOX_DEG, 2)]
        out = cache / f"{f.stem}.json"
        if out.exists():
            try:
                c = json.loads(out.read_text())
                b = c["box"]
                if (c.get("version") == VERSION and c["sts"] <= sts and c["ets"] >= ets
                        and b[0] <= box[0] and b[1] >= box[1] and b[2] <= box[2] and b[3] >= box[3]):
                    continue
            except Exception:  # noqa: BLE001
                pass
        try:
            prods = fetch_sbw(sts, ets, box) + fetch_sps(sts, ets, box)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"{f.stem}: download failed ({e}); retrying next run")
            continue
        prods.sort(key=lambda p: (p["iss"] or 0, p["k"]))
        out.write_text(json.dumps({"version": VERSION, "sts": int(sts), "ets": int(ets), "box": box, "products": prods},
                                  separators=(",", ":")))
        n_new += 1
        cnt = pd.Series([p["k"] for p in prods]).value_counts().to_dict() if prods else {}
        print(f"{f.stem}: {len(prods)} products in the box {cnt}")
    print(f"NWS products: {n_new} events downloaded, {n_fail} failed, {len(files)} event files")


# ---------------------------------------------------------------- association
def _near(lat: np.ndarray, lon: np.ndarray, ring: list[list[float]], km: float) -> np.ndarray:
    """Points inside the ring or within km of its edge (local equirectangular km)."""
    r = np.asarray(ring, float)
    la0 = r[:, 0].mean()
    kx = 111.2 * np.cos(np.radians(la0))
    px, py = (lon - r[0, 1]) * kx, (lat - r[0, 0]) * 111.2
    vx, vy = (r[:, 1] - r[0, 1]) * kx, (r[:, 0] - r[0, 0]) * 111.2
    x0, y0, x1, y1 = vx[:-1], vy[:-1], vx[1:], vy[1:]
    P = px[:, None], py[:, None]
    # ray casting
    cond = (y0 > P[1]) != (y1 > P[1])
    xin = x0 + (P[1] - y0) * (x1 - x0) / np.where(y1 == y0, 1e-12, y1 - y0)
    inside = (cond & (P[0] < xin)).sum(1) % 2 == 1
    # distance to edges
    dx, dy = x1 - x0, y1 - y0
    L2 = np.where(dx * dx + dy * dy == 0, 1e-12, dx * dx + dy * dy)
    u = np.clip(((P[0] - x0) * dx + (P[1] - y0) * dy) / L2, 0, 1)
    d = np.hypot(P[0] - (x0 + u * dx), P[1] - (y0 + u * dy)).min(1)
    return inside | (d <= km)


def associate(products: list[dict], mesos: list[tuple[int, np.ndarray]]) -> list[dict]:
    """mesos: [(meso index, array [[t, lat, lon], ...] sorted by t)]. Returns products with "m" (meso indices)."""
    out = []
    for p in products:
        if p["iss"] is None:
            continue
        w0 = p["iss"] - ASSOC_BEFORE_S
        w1 = min(p["exp"] or p["iss"] + ASSOC_AFTER_S, p["iss"] + ASSOC_AFTER_S)
        hit = []
        for mi, pts in mesos:
            t0, t1 = max(w0, pts[0, 0]), min(w1, pts[-1, 0])
            if t0 > t1:
                continue
            ts = np.arange(t0, t1 + 1, 60.0) if t1 > t0 else np.array([t0])
            la, lo = np.interp(ts, pts[:, 0], pts[:, 1]), np.interp(ts, pts[:, 0], pts[:, 2])
            if any(_near(la, lo, ring, BUFFER_KM).any() for ring in p["poly"]):
                hit.append(mi)
        if hit:
            out.append({**p, "m": hit})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="tracks")
    ap.add_argument("--cache", default="nws-cache")
    ap.add_argument("--max-minutes", type=float, default=30)
    fetch_main(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())
