#!/usr/bin/env python3
"""Hourly near-storm environments from HRRR native levels (Coffer et al. 2025 "HRRR-Inflow").

For every mesocyclone in tracks/ and every top of the hour within its tracked lifetime
(or, if it never spans one, the hour before it formed):
  * position at HH:00 interpolated along its track; observed motion from track positions
    +/-15 min around HH:00 (clamped to the track);
  * HRRR 00-h analysis valid at HH:00, native hybrid levels (s3://noaa-hrrr-bdp-pds,
    hrrr.tHHz.wrfnatf00.grib2), winds rotated from grid- to earth-relative;
  * 90-degree slice (+/-45 deg) centered on the 0-1 km storm-relative inflow direction
    (observed motion; Bunkers right-mover if no observed motion) within 40 km of the grid point
    nearest the storm; theta, qv, p, u, v and height AGL averaged over the slice -> one sounding;
  * parameters from scripts/envparams.py.

Outputs:
  env-profiles/YYYYMMDD_<meso>_<HHMM>_profile.csv   (event date in the name; profile to 100 hPa)
  env-params/YYYYMMDD.csv                           (one row per meso per hour)
Results are cached in env-cache/cache.jsonl (keyed by valid hour, position and motion), so
each HRRR hour is processed once; a run stops starting new hours after --max-minutes.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import re
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent))

BUCKET = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
VARS = {"PRES": "p", "HGT": "z", "TMP": "t", "SPFH": "q", "UGRD": "u", "VGRD": "v"}
RADIUS_KM = 40.0
HALF_ANGLE = 45.0
WIN = 16            # grid points either side of the nearest point (3 km grid -> +/-48 km)
MOTION_WIN = 15 * 60
P_TOP = 100.0       # hPa, top of the written profiles
RD_CP = 0.2857

PARAM_COLS = (
    ["bunk_rm_u", "bunk_rm_v", "bunk_lm_u", "bunk_lm_v", "mw06_u", "mw06_v",
     "sbcape", "sbcin", "sblcl", "sblfc", "sbel",
     "mlcape", "mlcin", "mllcl", "mllfc", "mlel",
     "mucape", "mucin", "mulpl", "mulfc", "muel",
     "mlcape03", "ecape_obs", "ecape_bunk", "eff_base", "eff_top", "eff_depth", "dcape",
     "lr03", "lr36", "lr75", "t_sfc", "td_sfc", "mlmr", "pw", "rh01", "rh13", "rh36", "frz_lvl",
     "grw500", "grw1", "grw6", "grw05_3", "bwd500", "bwd1", "bwd3", "bwd6", "bwd13", "ebwd"]
    + [f"{n}_{t}" for n in ["srh500", "srh1", "srh3", "srh13", "esrh", "srf500", "srf1", "srf3", "srf13", "srf01",
                            "sw500", "cw500", "psw500", "crit_angle"] for t in ("obs", "bunk")]
    + [f"{n}_{t}" for n in ["stp_fixed", "stp_eff", "stp500", "scp"] for t in ("obs", "bunk")]
    + ["flag_mucape0", "flag_eib_elev", "flag_ebwd_lt12p5"]
)
META_COLS = ["meso_id", "date", "time_utc", "lat", "lon", "prev_hour", "slice_dir", "slice_npts",
             "obs_u", "obs_v", "obs_spd", "obs_dir"]


# ----------------------------------------------------------------------------- HRRR access
def http_get(url, headers=None, timeout=120):
    for attempt in range(5):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except Exception:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def inventory(valid: datetime):
    base = f"{BUCKET}/hrrr.{valid:%Y%m%d}/conus/hrrr.t{valid:%H}z.wrfnatf00.grib2"
    r = http_get(base + ".idx")
    if r is None:
        return base, None
    lines = [ln.split(":") for ln in r.text.strip().splitlines()]
    out = []
    for k, f in enumerate(lines):
        start = int(f[1])
        end = int(lines[k + 1][1]) - 1 if k + 1 < len(lines) else None
        out.append((f[3], f[4], start, end))
    return base, out


_GRID = {}


def grid_latlon(msg):
    if "lat" not in _GRID:
        lat, lon = msg.latlons()
        lon = np.where(lon > 180, lon - 360, lon)
        _GRID["lat"], _GRID["lon"] = lat, lon
        # local angle of the grid x-axis, counterclockwise from east (for wind rotation)
        dlon = np.gradient(lon, axis=1) * np.cos(np.radians(lat))
        dlat = np.gradient(lat, axis=1)
        _GRID["ang"] = np.arctan2(dlat, dlon)
    return _GRID["lat"], _GRID["lon"], _GRID["ang"]


def process_hour(valid_iso: str, storms: list[dict]) -> dict:
    """Download one HRRR native-level analysis and build inflow soundings for all storms."""
    import pygrib
    import envparams

    t0 = time.time()
    valid = datetime.fromisoformat(valid_iso)
    base, inv = inventory(valid)
    if inv is None:
        return {"valid": valid_iso, "error": "HRRR file not found", "results": {}}
    want = []
    for var, lev, s, e in inv:
        m = re.fullmatch(r"(\d+) hybrid level", lev)
        if var in VARS and m:
            want.append((var, int(m.group(1)), s, e))
        elif var == "HGT" and lev == "surface":
            want.append(("TERRAIN", 0, s, e))
    nlev = max(l for v, l, _, _ in want if v != "TERRAIN")
    have_terrain = any(v == "TERRAIN" for v, *_ in want)

    # windows per storm
    first = None
    cube = {}  # (storm idx) -> {var: [nlev, ny, nx]}
    nbytes = 0
    sess = requests.Session()

    def fetch(item):
        var, lev, s, e = item
        rng = f"bytes={s}-{e}" if e is not None else f"bytes={s}-"
        for attempt in range(5):
            try:
                r = sess.get(base, headers={"Range": rng}, timeout=120)
                r.raise_for_status()
                return item, r.content
            except Exception:  # noqa: BLE001
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))

    win_idx = None
    terrain_win = {}
    with cf.ThreadPoolExecutor(8) as tp:
        for (var, lev, _, _), raw in tp.map(fetch, want):
            nbytes += len(raw)
            msg = pygrib.fromstring(raw)
            if first is None:
                first = msg
                lat, lon, ang = grid_latlon(msg)
                win_idx = []
                for st in storms:
                    d2 = (lat - st["lat"]) ** 2 + ((lon - st["lon"]) * math.cos(math.radians(st["lat"]))) ** 2
                    j, i = np.unravel_index(np.argmin(d2), d2.shape)
                    j0, j1 = max(0, j - WIN), min(lat.shape[0], j + WIN + 1)
                    i0, i1 = max(0, i - WIN), min(lat.shape[1], i + WIN + 1)
                    win_idx.append((j0, j1, i0, i1, j - j0, i - i0))
            vals = np.asarray(msg.values, dtype=np.float64)
            for k, (j0, j1, i0, i1, _, _) in enumerate(win_idx):
                w = vals[j0:j1, i0:i1]
                if var == "TERRAIN":
                    terrain_win[k] = w
                    continue
                c = cube.setdefault(k, {})
                arr = c.setdefault(VARS[var], np.full((nlev,) + w.shape, np.nan))
                arr[lev - 1] = w

    lat, lon, ang = grid_latlon(first)
    results = {}
    for k, st in enumerate(storms):
        try:
            j0, j1, i0, i1, cj, ci = win_idx[k]
            c = cube[k]
            a = ang[j0:j1, i0:i1]
            ug, vg = c["u"], c["v"]
            ue = ug * np.cos(a) - vg * np.sin(a)          # grid-relative -> earth-relative
            ve = ug * np.sin(a) + vg * np.cos(a)
            terr = terrain_win.get(k)
            if terr is None:                               # fallback: lowest level ~ 8 m AGL
                terr = c["z"][0] - 8.0
            zagl = c["z"] - terr[None]
            # motion for the slice: observed if available, else Bunkers RM at the nearest point
            P0, T0, Q0 = c["p"][:, cj, ci] / 100, c["t"][:, cj, ci] - 273.15, c["q"][:, cj, ci]
            U0, V0, Z0 = ue[:, cj, ci], ve[:, cj, ci], zagl[:, cj, ci]
            mot = st.get("motion")
            if mot is None:
                import metpy.calc as mpcalc
                from metpy.units import units
                rm, _, _ = mpcalc.bunkers_storm_motion(P0 * units.hPa, U0 * units("m/s"), V0 * units("m/s"), Z0 * units.m)
                mot_slice = (float(rm[0].m), float(rm[1].m))
            else:
                mot_slice = mot
            sel01 = Z0 <= 1000
            sru, srv = U0[sel01].mean() - mot_slice[0], V0[sel01].mean() - mot_slice[1]
            inflow_az = (math.degrees(math.atan2(-sru, -srv)) + 360) % 360   # bearing toward the inflow source
            # slice points
            la, lo = lat[j0:j1, i0:i1], lon[j0:j1, i0:i1]
            p1, p2 = math.radians(st["lat"]), np.radians(la)
            dl = np.radians(lo - st["lon"])
            dist = 6371.0 * 2 * np.arcsin(np.sqrt(np.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2))
            brg = (np.degrees(np.arctan2(np.sin(dl) * np.cos(p2), math.cos(p1) * np.sin(p2) - math.sin(p1) * np.cos(p2) * np.cos(dl))) + 360) % 360
            diff = np.abs((brg - inflow_az + 180) % 360 - 180)
            mask = (dist <= RADIUS_KM) & ((diff <= HALF_ANGLE) | (dist < 1.5))
            npts = int(mask.sum())
            th = c["t"] * (100000.0 / c["p"]) ** RD_CP
            avg = lambda a3: np.array([lvl[mask].mean() for lvl in a3])
            p_m, th_m, q_m = avg(c["p"]), avg(th), avg(c["q"])
            u_m, v_m, z_m = avg(ue), avg(ve), avg(zagl)
            terr_m = float(np.asarray(terr)[mask].mean())
            t_m = th_m * (p_m / 100000.0) ** RD_CP - 273.15
            p_h = p_m / 100.0
            e = q_m * p_h / (0.622 + 0.378 * q_m)          # vapor pressure (hPa)
            td_m = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))
            params = envparams.compute(z_m, p_h, t_m, td_m, q_m, u_m, v_m, terr_m, mot)
            keep = p_h >= P_TOP
            prof = [[round(float(z_m[i]), 1), round(float(p_h[i]), 2), round(float(t_m[i]), 2),
                     round(float(td_m[i]), 2), round(float(u_m[i]), 2), round(float(v_m[i]), 2)]
                    for i in np.where(keep)[0]]
            params.update({"slice_dir": round(inflow_az, 1), "slice_npts": npts})
            results[st["key"]] = {"params": params, "profile": prof}
        except Exception as e:  # noqa: BLE001
            results[st["key"]] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}
    return {"valid": valid_iso, "results": results, "bytes": nbytes, "secs": round(time.time() - t0, 1),
            "nlev": nlev, "terrain": have_terrain}


# ----------------------------------------------------------------------------- tracks -> storm-hours
def natural(sid):
    return (len(sid), sid)


def read_tracks(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    lines = text.split("\n")
    split = next((i for i, ln in enumerate(lines) if ln.strip().strip(",").lower() == "tornadoes"), len(lines))
    import io
    df = pd.read_csv(io.StringIO("\n".join(ln for ln in lines[:split] if ln.strip().strip(","))), dtype=str, keep_default_na=False)
    df["_t"] = pd.to_datetime(df["time_utc"], utc=True, errors="coerce", format="mixed")
    df["_la"] = pd.to_numeric(df["lat"], errors="coerce")
    df["_lo"] = pd.to_numeric(df["lon"], errors="coerce")
    return df.dropna(subset=["_t", "_la", "_lo"])


def storm_hours(path: Path) -> list[dict]:
    df = read_tracks(path)
    out = []
    for (sc, ms), g in df.groupby(["supercell_id", "meso_id"]):
        g = g.sort_values("_t")
        ts = np.array([x.timestamp() for x in g["_t"]])
        la, lo = g["_la"].to_numpy(float), g["_lo"].to_numpy(float)
        t_first, t_last = ts[0], ts[-1]
        first_h = math.ceil(t_first / 3600) * 3600
        hours = list(range(int(first_h), int(t_last) + 1, 3600))
        prev = False
        if not hours:
            hours = [int(math.floor(t_first / 3600) * 3600)]
            prev = True
        pos = lambda t: (float(np.interp(t, ts, la)), float(np.interp(t, ts, lo)))
        for h in hours:
            plat, plon = pos(min(max(h, t_first), t_last))
            if prev:
                a, b = t_first, t_last
            else:
                a, b = max(t_first, h - MOTION_WIN), min(t_last, h + MOTION_WIN)
            motion = None
            if b - a >= 300:
                (la1, lo1), (la2, lo2) = pos(a), pos(b)
                dx = (lo2 - lo1) * 111320.0 * math.cos(math.radians((la1 + la2) / 2))
                dy = (la2 - la1) * 110540.0
                motion = (round(dx / (b - a), 2), round(dy / (b - a), 2))
            valid = datetime.fromtimestamp(h, timezone.utc)
            mkey = "na" if motion is None else f"{motion[0]:.1f},{motion[1]:.1f}"
            out.append({
                "meso_id": f"{sc}{ms}", "sc": sc, "ms": int(ms), "valid": valid, "lat": round(plat, 4), "lon": round(plon, 4),
                "prev_hour": "yes" if prev else "", "motion": motion,
                "key": f"{valid:%Y%m%d%H}|{plat:.3f}|{plon:.3f}|{mkey}",
            })
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", default="tracks")
    ap.add_argument("--profiles", default="env-profiles")
    ap.add_argument("--params", default="env-params")
    ap.add_argument("--cache", default="env-cache/cache.jsonl")
    ap.add_argument("--status", default="env-cache/status.json")
    ap.add_argument("--max-minutes", type=float, default=200)
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 2)))
    a = ap.parse_args()
    started = time.time()
    status = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "errors": []}
    Path(a.cache).parent.mkdir(parents=True, exist_ok=True)

    cache = {}
    if Path(a.cache).exists():
        for ln in Path(a.cache).read_text().splitlines():
            if ln.strip():
                rec = json.loads(ln)
                cache[rec["key"]] = rec
    n0 = len(cache)

    events = {}
    for p in sorted(Path(a.tracks).glob("*.csv")):
        if p.name == "id_map.csv":
            continue
        try:
            events[p.stem] = storm_hours(p)
        except Exception as e:  # noqa: BLE001
            status["errors"].append(f"{p.name}: {e}")

    # group uncached storm-hours by valid hour
    todo: dict[str, list] = {}
    for rows in events.values():
        for r in rows:
            if r["key"] not in cache:
                lst = todo.setdefault(r["valid"].isoformat(), [])
                if all(x["key"] != r["key"] for x in lst):
                    lst.append({"key": r["key"], "lat": r["lat"], "lon": r["lon"], "motion": r["motion"]})
    hours = sorted(todo)
    status["hours_needed"] = len(hours)
    status["storm_hours_needed"] = sum(len(v) for v in todo.values())

    done = failed = nbytes = 0
    secs, info = [], {}
    deadline = started + a.max_minutes * 60
    pending = list(hours)
    with cf.ProcessPoolExecutor(max_workers=a.workers) as ex:
        running = {}
        while pending or running:
            while pending and len(running) < a.workers and time.time() < deadline:
                h = pending.pop(0)
                running[ex.submit(process_hour, h, todo[h])] = h
            if not running:
                break
            fin, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
            for f in fin:
                h = running.pop(f)
                try:
                    res = f.result()
                    if res.get("error"):
                        failed += 1
                        status["errors"].append(f"{h}: {res['error']}")
                        continue
                    for key, rr in res["results"].items():
                        if "error" in rr:
                            if len(status["errors"]) < 30:
                                status["errors"].append(f"{h} {key}: {rr['error']} | {rr.get('trace', '')}")
                            continue
                        cache[key] = {"key": key, **rr}
                    done += 1
                    nbytes += res["bytes"]
                    secs.append(res["secs"])
                    info = {"levels": res["nlev"], "terrain_field": res["terrain"]}
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    if len(status["errors"]) < 30:
                        status["errors"].append(f"{h}: {type(e).__name__}: {e}")
        if pending:
            status["stopped_early"] = f"time budget ({a.max_minutes} min) reached"

    # ---- write outputs
    prof_dir, par_dir = Path(a.profiles), Path(a.params)
    prof_dir.mkdir(parents=True, exist_ok=True)
    par_dir.mkdir(parents=True, exist_ok=True)
    for old in list(prof_dir.glob("*.csv")) + list(par_dir.glob("*.csv")):
        old.unlink()
    n_rows = n_missing = 0
    for ev, rows in events.items():
        out_rows = []
        for r in sorted(rows, key=lambda r: (natural(r["sc"]), r["ms"], r["valid"])):
            rec = cache.get(r["key"])
            mot = r["motion"]
            meta = {
                "meso_id": r["meso_id"], "date": f"{r['valid']:%Y-%m-%d}", "time_utc": f"{r['valid']:%H%M}",
                "lat": r["lat"], "lon": r["lon"], "prev_hour": r["prev_hour"],
                "obs_u": mot[0] if mot else "", "obs_v": mot[1] if mot else "",
                "obs_spd": round(math.hypot(*mot), 2) if mot else "",
                "obs_dir": round((math.degrees(math.atan2(mot[0], mot[1])) + 360) % 360, 1) if mot else "",
            }
            if rec is None:
                n_missing += 1
                out_rows.append(meta)
                continue
            prm = rec["params"]
            row = {**meta, "slice_dir": prm.get("slice_dir", ""), "slice_npts": prm.get("slice_npts", "")}
            for c in PARAM_COLS:
                v = prm.get(c, "")
                row[c] = "" if (isinstance(v, float) and v != v) else (round(v, 4) if isinstance(v, float) else v)
            out_rows.append(row)
            n_rows += 1
            # profile file
            fn = prof_dir / f"{ev}_{r['meso_id']}_{r['valid']:%H%M}_profile.csv"
            head = (f"event_date={ev},valid_time_utc={r['valid']:%Y-%m-%dT%H:%MZ},meso_id={r['meso_id']},"
                    f"lat={r['lat']},lon={r['lon']}\n")
            body = "height_agl_m,pressure_hpa,temperature_c,dewpoint_c,u_ms,v_ms\n" + "".join(
                ",".join(str(x) for x in lv) + "\n" for lv in rec["profile"])
            fn.write_text(head + body)
        pd.DataFrame(out_rows, columns=META_COLS + PARAM_COLS).to_csv(par_dir / f"{ev}.csv", index=False, lineterminator="\n")

    with open(a.cache, "w") as fh:
        for k in sorted(cache):
            fh.write(json.dumps(cache[k], separators=(",", ":")) + "\n")
    status.update({
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "minutes": round((time.time() - started) / 60, 1),
        "events": len(events), "hours_done": done, "hours_failed": failed, "hours_left": len(pending),
        "downloaded_gb": round(nbytes / 1e9, 2), "secs_per_hour_median": float(np.median(secs)) if secs else None,
        "storm_hours_written": n_rows, "storm_hours_missing": n_missing,
        "cache_entries": len(cache), "new_cache_entries": len(cache) - n0, **info,
    })
    Path(a.status).write_text(json.dumps(status, indent=1))
    print(json.dumps({k: v for k, v in status.items() if k != "errors"}, indent=1), "\nerrors:", status["errors"][:5])


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        Path("env-cache").mkdir(exist_ok=True)
        Path("env-cache/status.json").write_text(json.dumps({"fatal": traceback.format_exc()}, indent=1))
        raise
