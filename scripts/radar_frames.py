#!/usr/bin/env python3
"""Lowest-tilt reflectivity loop for one event: closest-radar mosaic near the active storms.

Every STEP_S (5 min) from 30 min before the event's first track point to 30 min after its last, the frame
covers only the ground within CROP_KM of a mesocyclone active at that time (its track spans the time,
padded by ACTIVE_PAD_S). Each pixel (~1 km, on a Web Mercator grid so the page can drop it on the map as
is) takes the lowest-tilt (0.5 deg) surveillance reflectivity of its closest NEXRAD, from that radar's most
recent low sweep (SAILS sweeps included) no older than MAX_AGE_S. Volumes come from the NEXRAD Level II
archive on AWS (needs network: run in GitHub Actions, radar-frames.yml).

Output: <out>/<event>/index.json and one paletted PNG per frame (1 dB steps from 5 dBZ; transparent
below 5 dBZ and where no radar data).

  python scripts/radar_frames.py --event 20230331 --tracks tracks --out radar --workers 4
  python scripts/radar_frames.py --plan --tracks tracks --manifest radar-cache/manifest.json
      -> JSON {"build": [{"event", "fp"}...], "drop": [events]}: events whose track points changed (or that
         have no loop yet), and loops whose event file is gone. radar-frames.yml uses this every night.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "ml"))

VERSION = 2             # bump to rebuild every loop after a change to how frames are made
STEP_S = 300
PAD_S = 1800            # frames from 30 min before the first to 30 min after the last track point
ACTIVE_PAD_S = 600      # a meso counts as active from 10 min before its first to 10 min after its last point
CROP_KM = 50.0
DX_KM = 1.0
N_RANK = 3              # if the closest radar has no sweep for a frame, the 2nd (then 3rd) closest fills in
MAX_AGE_S = 720         # a sweep older than 12 min is not used
SITE_SEARCH_KM = 460
DBZ_MIN, DBZ_MAX = 5, 80
R_EARTH = 6378137.0     # Web Mercator sphere


# ---------------------------------------------------------------- color table (NWS-like reflectivity)
_STOPS = [(5, (4, 233, 231)), (10, (1, 159, 244)), (15, (3, 0, 244)), (20, (2, 253, 2)), (25, (1, 197, 1)),
          (30, (0, 142, 0)), (35, (253, 248, 2)), (40, (229, 188, 0)), (45, (253, 149, 0)), (50, (253, 0, 0)),
          (55, (212, 0, 0)), (60, (188, 0, 0)), (65, (248, 0, 253)), (70, (152, 84, 198)), (75, (253, 253, 253)),
          (80, (253, 253, 253))]


def palette() -> list[int]:
    """Index 0 transparent; index k (1..76) = DBZ_MIN + k - 1 dBZ."""
    xs = [s for s, _ in _STOPS]
    pal = [0, 0, 0]
    for z in range(DBZ_MIN, DBZ_MAX + 1):
        # stepwise within each 5-dB band (like the NWS table), brightening slightly toward the band top
        j = max(i for i, x in enumerate(xs) if x <= z)
        c0 = np.array(_STOPS[j][1], float)
        c1 = np.array(_STOPS[min(j + 1, len(xs) - 1)][1], float)
        f = (z - xs[j]) / 5.0 * 0.35
        pal += [int(round(v)) for v in c0 + (c1 - c0) * f]
    return pal + [0, 0, 0] * (256 - len(pal) // 3)


# ---------------------------------------------------------------- geometry
def merc(lat, lon):
    return R_EARTH * np.radians(lon), R_EARTH * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def unmerc(x, y):
    return np.degrees(2 * np.arctan(np.exp(y / R_EARTH)) - np.pi / 2), np.degrees(x / R_EARTH)


def km(lat0, lon0, lat1, lon1):
    p0, p1 = np.radians(lat0), np.radians(lat1)
    a = np.sin((p1 - p0) / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(np.radians(lon1 - lon0) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def bearing(lat0, lon0, lat1, lon1):
    p0, p1, dl = np.radians(lat0), np.radians(lat1), np.radians(lon1 - lon0)
    y = np.sin(dl) * np.cos(p1)
    x = np.cos(p0) * np.sin(p1) - np.sin(p0) * np.cos(p1) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def slant_km(ground_km, elev_deg):
    """Slant range of a beam at elevation elev reaching ground distance s (4/3 earth)."""
    re = 4.0 / 3.0 * 6371.0
    e = np.radians(elev_deg)
    th = ground_km / re
    return re * np.sin(th) / np.cos(th + e)


# ---------------------------------------------------------------- tracks and frames
def meso_tracks(path: Path) -> list[np.ndarray]:
    import nws_products
    d = nws_products.read_points(path)
    return [g.sort_values("t")[["t", "la", "lo"]].to_numpy(float) for _, g in d.groupby(["supercell_id", "meso_id"])]


def plan(mesos: list[np.ndarray], sites: pd.DataFrame):
    t0 = min(m[0, 0] for m in mesos) - PAD_S
    t1 = max(m[-1, 0] for m in mesos) + PAD_S
    ts = np.arange(math.floor(t0 / STEP_S) * STEP_S, t1 + 1, STEP_S)
    lat_mid = float(np.mean([m[:, 1].mean() for m in mesos]))
    dx = DX_KM * 1000.0 / math.cos(math.radians(lat_mid))       # Mercator metres per ~1 km of ground
    lat_all = np.concatenate([m[:, 1] for m in mesos]); lon_all = np.concatenate([m[:, 2] for m in mesos])
    near = km(lat_all.mean(), lon_all.mean(), sites.lat.to_numpy(), sites.lon.to_numpy()) < \
        km(lat_all.min(), lon_all.min(), lat_all.max(), lon_all.max()) / 2 + SITE_SEARCH_KM
    st = sites[near].reset_index(drop=True)
    frames = []
    for i, t in enumerate(ts):
        pos = []
        for m in mesos:
            if m[0, 0] - ACTIVE_PAD_S <= t <= m[-1, 0] + ACTIVE_PAD_S:
                tt = min(max(t, m[0, 0]), m[-1, 0])
                pos.append((np.interp(tt, m[:, 0], m[:, 1]), np.interp(tt, m[:, 0], m[:, 2])))
        if not pos:
            continue
        pos = np.array(pos)
        X, Y = merc(pos[:, 0], pos[:, 1])
        rad = CROP_KM * 1000.0 / np.cos(np.radians(pos[:, 0]))
        c0, c1 = int(np.floor((X - rad).min() / dx)), int(np.ceil((X + rad).max() / dx))
        r0, r1 = int(np.floor((Y - rad).min() / dx)), int(np.ceil((Y + rad).max() / dx))
        cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        px, py = (cc + 0.5) * dx, (rr + 0.5) * dx
        la, lo = unmerc(px, py)
        mask = np.zeros(la.shape, bool)
        for (pla, plo) in pos:
            mask |= km(pla, plo, la, lo) <= CROP_KM
        la_m, lo_m = la[mask], lo[mask]
        d = km(la_m[:, None], lo_m[:, None], st.lat.to_numpy()[None], st.lon.to_numpy()[None])
        near_site = np.argsort(d, axis=1)[:, :N_RANK]             # closest, 2nd, 3rd closest radar
        frames.append({"i": i, "t": float(t), "c0": c0, "c1": c1, "r0": r0, "r1": r1,
                       "shape": mask.shape, "site": near_site, "lat": la_m, "lon": lo_m,
                       "flat_rows": np.nonzero(mask)})
    return ts, frames, st, dx


# ---------------------------------------------------------------- per-radar worker
def low_sweeps(path: Path):
    """[(sweep time epoch, elev, sorted az, gate km, Z float32 [naz, ngate])] for the lowest-tilt surveillance sweeps."""
    import pyart
    radar = pyart.io.read_nexrad_archive(str(path), include_fields=["reflectivity", "differential_reflectivity"])
    fa = np.round(radar.fixed_angle["data"].astype(float), 1)
    low = fa.min()
    units = radar.time["units"].split("since ")[-1].strip().rstrip("Z")
    base = pd.Timestamp(units).tz_localize("UTC").timestamp() if pd.Timestamp(units).tzinfo is None else pd.Timestamp(units).timestamp()
    rng = radar.range["data"] / 1000.0
    out = []
    have_zdr = "differential_reflectivity" in radar.fields
    for i in range(radar.nsweeps):
        if fa[i] > low + 0.25:
            continue
        sl = radar.get_slice(i)
        z = radar.fields["reflectivity"]["data"][sl]
        if np.ma.count(z) == 0:
            continue
        if have_zdr and np.ma.count(radar.fields["differential_reflectivity"]["data"][sl]) == 0:
            continue                                     # Doppler half of a split cut
        az = radar.azimuth["data"][sl]
        order = np.argsort(az)
        t = base + float(np.mean(radar.time["data"][sl]))
        out.append((t, float(fa[i]), az[order], rng, np.ma.filled(z.astype(np.float32), np.nan)[order]))
    return out


def sample_sweep(sw, lat0, lon0, la, lo):
    _, elev, az, rng, z = sw
    g_km = km(lat0, lon0, la, lo)
    b = bearing(lat0, lon0, la, lo)
    s = slant_km(g_km, elev)
    k = np.searchsorted(az, b)
    k0 = (k - 1) % len(az); k1 = k % len(az)
    d0 = np.abs((b - az[k0] + 180) % 360 - 180); d1 = np.abs((az[k1] - b + 180) % 360 - 180)
    ai = np.where(d0 <= d1, k0, k1)
    gate = np.round((s - rng[0]) / (rng[1] - rng[0])).astype(int)
    ok = (gate >= 0) & (gate < z.shape[1]) & (np.minimum(d0, d1) <= 1.0)
    v = np.full(len(la), np.nan, np.float32)
    v[ok] = z[ai[ok], gate[ok]]
    return v


def radar_job(job):
    """job = (site, lat, lon, [(frame k, t, lat array, lon array)]) -> (site, {frame k: uint8 codes}, log)."""
    import requests
    from l2_assign import list_keys
    site, slat, slon, needs = job
    needs.sort(key=lambda n: n[1])
    tmin, tmax = needs[0][1] - MAX_AGE_S, needs[-1][1]
    keys = {}
    d = datetime.fromtimestamp(tmin, timezone.utc).date()
    while d <= datetime.fromtimestamp(tmax, timezone.utc).date():
        keys.update(list_keys(site, datetime.combine(d, datetime.min.time())))
        d += timedelta(days=1)
    vt = np.array(sorted(keys))
    want = np.zeros(len(vt), bool)
    for _, t, _, _ in needs:
        want |= (vt <= t) & (vt >= t - MAX_AGE_S)
    sweeps, out, n_vol, n_fail, t_start = [], {}, 0, 0, time.time()
    vol_iter = iter([v for v in vt[want]])
    pending = next(vol_iter, None)
    with tempfile.TemporaryDirectory() as td:
        for k, t, la, lo in needs:
            while pending is not None and pending <= t:
                f = Path(td) / "vol"
                try:
                    r = None
                    for attempt in range(4):
                        try:
                            r = requests.get(keys[pending], timeout=180); r.raise_for_status(); break
                        except Exception:  # noqa: BLE001
                            if attempt == 3:
                                raise
                            time.sleep(5)
                    f.write_bytes(r.content)
                    sweeps += low_sweeps(f)
                    n_vol += 1
                except Exception:  # noqa: BLE001
                    n_fail += 1
                sweeps = [s for s in sweeps if s[0] >= t - 2 * MAX_AGE_S]
                pending = next(vol_iter, None)
            cand = [s for s in sweeps if t - MAX_AGE_S <= s[0] <= t + 60]
            if not cand:
                continue
            sw = max(cand, key=lambda s: s[0])
            v = sample_sweep(sw, slat, slon, la, lo)
            code = np.zeros(len(v), np.uint8)
            ok = np.isfinite(v) & (v >= DBZ_MIN - 0.5)
            code[ok] = (np.clip(np.round(v[ok]), DBZ_MIN, DBZ_MAX) - DBZ_MIN + 1).astype(np.uint8)
            out[k] = (code, sw[0])
    return site, out, f"{site}: {n_vol} volumes ({n_fail} failed), {len(out)}/{len(needs)} frames, {(time.time() - t_start) / 60:.1f} min"


def _init():
    import os
    os.environ["OMP_NUM_THREADS"] = "1"


def outline(rows, cols, r0, c0, dx, tol_px=1.0):
    """Boundary of a set of frame pixels as [[[lat, lon], ...], ...] rings (exterior and holes), simplified."""
    from shapely.geometry import box
    from shapely.ops import unary_union
    boxes = []
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    start = 0
    for i in range(1, len(rows) + 1):                 # merge each row's runs of pixels into one box
        if i == len(rows) or rows[i] != rows[start] or cols[i] != cols[i - 1] + 1:
            boxes.append(box(cols[start], rows[start], cols[i - 1] + 1, rows[start] + 1))
            start = i
    geom = unary_union(boxes).simplify(tol_px)
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    out = []
    for p in polys:
        if p.is_empty or p.area < 4:                   # ignore specks under ~4 km^2
            continue
        for ring in [p.exterior, *p.interiors]:
            xy = np.asarray(ring.coords)
            la, lo = unmerc((xy[:, 0] + c0) * dx, (xy[:, 1] + r0) * dx)
            out.append([[round(float(a), 3), round(float(b), 3)] for a, b in zip(la, lo)])
    return out


# ---------------------------------------------------------------- what needs (re)building
def fingerprint(path: Path) -> str | None:
    """Hash of the track points (time, position) and VERSION: only these change the frames."""
    import hashlib
    import nws_products
    try:
        d = nws_products.read_points(path)
    except Exception:  # noqa: BLE001
        return None
    if d.empty:
        return None
    pts = sorted(zip(d.t.round().astype(int), d.la.round(4), d.lo.round(4)))
    return hashlib.sha1(f"v{VERSION}|{pts}".encode()).hexdigest()[:16]


def plan_main(tracks: Path, manifest: Path) -> None:
    done = json.loads(manifest.read_text()) if manifest.exists() else {}
    events = {p.stem: p for p in sorted(tracks.glob("*.csv")) if p.stem[:8].isdigit() and len(p.stem) == 8}
    build = []
    for ev, p in events.items():
        fp = fingerprint(p)
        if fp and done.get(ev) != fp:
            build.append({"event": ev, "fp": fp})
    drop = sorted(set(done) - set(events))
    print(json.dumps({"build": build, "drop": drop}))


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--manifest", default="radar-cache/manifest.json")
    ap.add_argument("--tracks", default="tracks")
    ap.add_argument("--out", default="radar")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.plan:
        return plan_main(Path(a.tracks), Path(a.manifest))
    from PIL import Image
    t_start = time.time()
    mesos = meso_tracks(Path(a.tracks) / f"{a.event}.csv")
    sites = pd.read_csv(ROOT / "ml" / "l2" / "nexrad_sites.csv")
    sites = sites[sites.site.str.startswith("K")]
    ts, frames, st, dx = plan(mesos, sites)
    print(f"{a.event}: {len(mesos)} mesos, {len(frames)} frames, {sum(len(f['lat']) for f in frames) / 1e6:.2f} M pixels, "
          f"{len({s for f in frames for s in np.unique(f['site'][:, 0])})} radars", flush=True)
    # Fill each pixel from its closest radar; where that radar has no usable sweep for the frame (missing from
    # the archive, a download failure, or older than MAX_AGE_S), from the next closest; up to N_RANK radars.
    for f in frames:
        f["lvl"] = np.zeros(len(f["lat"]), np.int8)
        f["code"] = np.zeros(len(f["lat"]), np.uint8)
        f["done"] = np.zeros(len(f["lat"]), bool)
        f["sweep_t"] = {}
    failed: set = set()
    n_pass = 0
    while True:
        jobs = {}
        for k, f in enumerate(frames):
            while True:                                   # move pixels off radars known to have nothing for k
                cur = f["site"][np.arange(len(f["lvl"])), f["lvl"]]
                bad = ~f["done"] & np.array([(int(si), k) in failed for si in cur]) if len(cur) else np.zeros(0, bool)
                if not bad.any():
                    break
                last = bad & (f["lvl"] >= N_RANK - 1)
                f["done"][last] = True                    # no radar at all: stays transparent
                f["lvl"][bad & ~last] += 1
            todo = ~f["done"]
            for si in np.unique(cur[todo]):
                sel = np.flatnonzero(todo & (cur == si))
                jobs.setdefault(int(si), []).append((k, f["t"], f["lat"][sel], f["lon"][sel], sel))
        if not jobs:
            break
        n_pass += 1
        if n_pass > 1:
            print(f" pass {n_pass} (fallback to the next closest radar): {sum(len(v) for v in jobs.values())} radar-frames", flush=True)
        with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
            args = [(st.site[si], st.lat[si], st.lon[si], [n[:4] for n in need]) for si, need in jobs.items()]
            for (si, need), (site, out, log) in zip(jobs.items(), ex.map(radar_job, args)):
                print(" ", log, flush=True)
                for k, _, _, _, sel in need:
                    r = out.get(k)
                    if r is None:
                        failed.add((si, k))
                        continue
                    f = frames[k]
                    f["code"][sel] = r[0]; f["done"][sel] = True
                    f["sweep_t"][site] = r[1]
    out_dir = Path(a.out) / a.event
    out_dir.mkdir(parents=True, exist_ok=True)
    pal = palette()
    idx, total, n_fill = [], 0, 0
    for k, f in enumerate(frames):
        h, w = f["shape"]
        img = np.zeros((h, w), np.uint8)
        rows, cols = f["flat_rows"]
        img[rows, cols] = f["code"]
        im = Image.fromarray(img[::-1], "P")                     # row 0 = north
        im.putpalette(pal)
        name = f"{int(f['t'])}.png"
        im.save(out_dir / name, optimize=True, transparency=0)
        size = (out_dir / name).stat().st_size
        total += size
        x0, x1, y0, y1 = f["c0"] * dx, f["c1"] * dx, f["r0"] * dx, f["r1"] * dx
        s_, w_ = (float(v) for v in unmerc(x0, y0))
        n_, e_ = (float(v) for v in unmerc(x1, y1))
        used = sorted(f["sweep_t"])
        rec = {"t": int(f["t"]), "f": name, "b": [[round(s_, 5), round(w_, 5)], [round(n_, 5), round(e_, 5)]],
               "radars": used, "age_s": int(max(f["t"] - v for v in f["sweep_t"].values())) if used else None,
               "px": int(len(f["lat"])), "bytes": size}
        # where a radar other than the closest was used: outline (lat/lon rings) and which radars stood in
        fb = f["done"] & (f["lvl"] > 0)
        if fb.any():
            n_fill += 1
            sub = f["site"][np.arange(len(f["lvl"])), f["lvl"]]
            rec["missing"] = sorted({st.site[int(si)] for si in f["site"][fb, 0]})
            rec["fill_by"] = sorted({st.site[int(si)] for si in sub[fb]})
            rec["fill"] = outline(rows[fb], cols[fb], f["r0"], f["c0"], dx)
        idx.append(rec)
    if n_fill:
        print(f" {n_fill} frames use a farther radar where the closest had no data", flush=True)
    (out_dir / "index.json").write_text(json.dumps({"event": a.event, "step_s": STEP_S, "crop_km": CROP_KM, "dx_km": DX_KM,
                                                     "dbz_min": DBZ_MIN, "frames": idx}, separators=(",", ":")))
    px = sum(f["px"] for f in idx)
    print(f"{a.event}: {len(idx)} frames, {total / 1e6:.2f} MB ({total / max(1, len(idx)) / 1e3:.0f} KB/frame, "
          f"{total / max(1, px) * 1e3:.1f} bytes per 1000 km^2); {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
