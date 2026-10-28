#!/usr/bin/env python3
"""Build site/data.json from MesoTrack daily CSVs + NCEI Storm Events tornado records.

MesoTrack CSV layout (one file per convective day, e.g. 20230331.csv):

    supercell_id,meso_id,time_utc,lat,lon,dissipation,primary_sup
    A,1,2023-03-31T17:45:42+00:00,33.92533,-94.29710,dissipate,
    ...
    <blank line>
    Tornadoes
    A1,1089126          <- supercell A, meso 1, NCEI EVENT_ID
    ...

Usage:
    python scripts/build_data.py --raw data/raw --out site/data.json
    python scripts/build_data.py --raw data/raw --ncei-local tests/ncei   # offline test
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ncei  # noqa: E402
import tordb  # noqa: E402


def clean_id(v) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()


def parse_file(path: Path) -> tuple[pd.DataFrame, list[tuple[str, str]], str]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    lines = [ln.rstrip("\r") for ln in lines]
    split = next((i for i, ln in enumerate(lines) if ln.strip().strip(",").lower() == "tornadoes"), len(lines))
    track_txt = "\n".join(ln for ln in lines[:split] if ln.strip().strip(","))
    df = pd.read_csv(io.StringIO(track_txt), dtype={"supercell_id": str, "meso_id": str, "primary_sup": str})
    df.columns = [c.strip().lower() for c in df.columns]
    links = []
    for ln in lines[split + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 1 and parts[0]:
            links.append((parts[0], parts[1] if len(parts) > 1 else ""))
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", path.stem)
    if m:
        day = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    else:
        day = pd.to_datetime(df["time_utc"].iloc[0], utc=True).strftime("%Y-%m-%d")
    return df, links, day


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="site/data.json")
    ap.add_argument("--ncei-cache", default="data/ncei")
    ap.add_argument("--ncei-local", default=None, help="folder of pre-downloaded NCEI details files (testing)")
    ap.add_argument("--tordb-dir", default=None, help="extra folder to search for the tornado database CSV")
    ap.add_argument("--azshear-dir", default=None, help="folder of MRMS AzShear series (tracks-AzShear/)")
    ap.add_argument("--env-params", default=None, help="folder of HRRR inflow parameters (env-params/)")
    ap.add_argument("--env-profiles", default=None, help="folder of HRRR inflow soundings (env-profiles/)")
    a = ap.parse_args()

    allcsv = sorted(p for p in Path(a.raw).rglob("*") if p.suffix.lower() == ".csv" and p.name != "id_map.csv")
    db_files = [p for p in allcsv if tordb.is_tordb(p)]
    files = [p for p in allcsv if p not in db_files]
    if a.tordb_dir:
        db_files += [p for p in sorted(Path(a.tordb_dir).rglob("*.csv")) if tordb.is_tordb(p) and p not in db_files]
    # original (pre-renumbering) IDs, if the files came from scripts/renumber.py
    orig = {}
    idmap = Path(a.raw) / "id_map.csv"
    if idmap.exists():
        im = pd.read_csv(idmap, dtype=str, keep_default_na=False)
        for r in im.itertuples():
            orig[(r.file, r.new_supercell, r.new_meso)] = (r.old_supercell, r.old_meso)
    if not files:
        sys.exit(f"No MesoTrack CSVs in {a.raw}")
    db = tordb.TorDB(db_files) if db_files else None
    if db is None:
        print("! No tornado database CSV (with a oneTorID column) found; using NCEI segments only")

    cases, supercells, mesos, tornadoes = [], [], [], []
    pending_links = []   # (meso_idx, event_id, case_label)
    problems = []

    for path in files:
        try:
            df, links, day = parse_file(path)
        except Exception as e:  # noqa: BLE001
            problems.append(f"{path.name}: unreadable ({e})")
            continue
        need = {"supercell_id", "meso_id", "time_utc", "lat", "lon"}
        if not need <= set(df.columns):
            problems.append(f"{path.name}: missing columns {sorted(need - set(df.columns))}")
            continue
        df["t"] = (pd.to_datetime(df["time_utc"], utc=True, errors="coerce", format="mixed")
                   - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
        df = df.dropna(subset=["t", "lat", "lon"])
        df["supercell_id"] = df["supercell_id"].map(clean_id)
        df["meso_id"] = df["meso_id"].map(clean_id)
        df = df.sort_values(["supercell_id", "meso_id", "t"])

        case_idx = len(cases)
        cases.append({"date": day, "file": path.name, "sc": []})
        sc_index, meso_lookup = {}, {}
        for (sc, mid), g in df.groupby(["supercell_id", "meso_id"], sort=True):
            if sc not in sc_index:
                sc_index[sc] = len(supercells)
                cases[case_idx]["sc"].append(len(supercells))
                o = orig.get((path.name, sc, mid))
                supercells.append({"case": case_idx, "label": sc, "mesos": [], "merged_into": None,
                                   "orig": o[0] if o and o[0] != sc else None})
            si = sc_index[sc]
            mi = len(mesos)
            end = str(g["dissipation"].dropna().iloc[-1]).strip() if "dissipation" in g and g["dissipation"].notna().any() else ""
            prim = str(g["primary_sup"].dropna().iloc[-1]).strip() if "primary_sup" in g and g["primary_sup"].notna().any() else ""
            if prim:
                supercells[si]["merged_into"] = prim
            mesos.append({
                "sc": si, "label": f"{sc}{mid}", "end": end,
                "orig": (f"{orig[(path.name, sc, mid)][0]}{orig[(path.name, sc, mid)][1]}"
                         if (path.name, sc, mid) in orig and orig[(path.name, sc, mid)] != (sc, mid) else None),
                "pts": [[int(r.t), round(float(r.lat), 4), round(float(r.lon), 4)] for r in g.itertuples()],
                "tor": [],
            })
            supercells[si]["mesos"].append(mi)
            meso_lookup[f"{sc}{mid}".upper()] = mi
        for key, ev in links:
            mi = meso_lookup.get(key.upper().replace(" ", ""))
            if mi is None:
                problems.append(f"{path.name}: tornado link '{key}' doesn't match any meso")
                continue
            pending_links.append((mi, ev, path.name))

    # MRMS AzShear series per meso: [[epoch, lat, lon, low, mid], ...] (values in 1e-3 s^-1)
    n_az = 0
    if a.azshear_dir and Path(a.azshear_dir).is_dir():
        lookup = {}
        for mi, m in enumerate(mesos):
            s_ = supercells[m["sc"]]
            lookup[(cases[s_["case"]]["file"], s_["label"], m["label"][len(s_["label"]):])] = mi
        for f in sorted(Path(a.azshear_dir).glob("*.csv")):
            try:
                az = pd.read_csv(f, dtype={"supercell_id": str, "meso_id": str})
            except Exception as e:  # noqa: BLE001
                problems.append(f"{f.name}: AzShear file unreadable ({e})")
                continue
            az["t"] = (pd.to_datetime(az["time_utc"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
            for (sc, ms), g in az.groupby(["supercell_id", "meso_id"]):
                mi = lookup.get((f.name, str(sc).strip(), str(ms).strip()))
                if mi is None:
                    continue
                num = lambda v: None if pd.isna(v) else round(float(v), 3)
                mesos[mi]["az"] = [[int(r.t), round(float(r.lat), 4), round(float(r.lon), 4), num(r.azshear_0_2km), num(r.azshear_3_6km)]
                                   for r in g.sort_values("t").itertuples()]
                n_az += 1
        print(f"AzShear series attached to {n_az} of {len(mesos)} mesocyclones")

    # HRRR inflow environments -> site/env.json (loaded by the page only when "Environments" is on)
    if a.env_params and Path(a.env_params).is_dir():
        lookup = {}
        for mi, m in enumerate(mesos):
            s_ = supercells[m["sc"]]
            lookup[(cases[s_["case"]]["file"][:8], m["label"].upper())] = mi
        def fnum(v, nd=1):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return None
            return None if v != v else round(v, nd)
        env = []
        for f in sorted(Path(a.env_params).glob("*.csv")):
            event = f.stem
            try:
                ep = pd.read_csv(f, dtype={"meso_id": str, "time_utc": str})
            except Exception as e:  # noqa: BLE001
                problems.append(f"{f.name}: env-params file unreadable ({e})")
                continue
            for r in ep.to_dict("records"):
                mid = str(r["meso_id"]).strip().upper()
                mi = lookup.get((event, mid))
                if mi is None:
                    continue
                hhmm = str(r["time_utc"]).zfill(4)
                t = int(pd.Timestamp(f"{r['date']}T{hhmm[:2]}:{hhmm[2:]}Z").timestamp())
                prof = []
                pf = Path(a.env_profiles or "") / f"{event}_{mid}_{hhmm}_profile.csv"
                if a.env_profiles and pf.exists():
                    try:
                        pr = pd.read_csv(pf, skiprows=1)
                        prof = [[fnum(x.height_agl_m, 0), fnum(x.pressure_hpa, 1), fnum(x.temperature_c, 1),
                                 fnum(x.dewpoint_c, 1), fnum(x.u_ms, 1), fnum(x.v_ms, 1)] for x in pr.itertuples()]
                    except Exception as e:  # noqa: BLE001
                        problems.append(f"{pf.name}: profile unreadable ({e})")
                env.append({
                    "m": mi, "t": t, "lat": fnum(r["lat"], 4), "lon": fnum(r["lon"], 4), "dir": fnum(r["slice_dir"], 1),
                    "prev": str(r.get("prev_hour", "")).strip().lower() == "yes",
                    "T": fnum(r.get("t_sfc")), "Td": fnum(r.get("td_sfc")),
                    "mlcape": fnum(r.get("mlcape"), 0), "mlcin": fnum(r.get("mlcin"), 0), "mllcl": fnum(r.get("mllcl"), 0), "mllfc": fnum(r.get("mllfc"), 0),
                    "eb": fnum(r.get("eff_base"), 0), "et": fnum(r.get("eff_top"), 0),
                    "esrh": fnum(r.get("esrh_obs"), 0), "esrh_b": fnum(r.get("esrh_bunk"), 0),
                    "obs": [fnum(r.get("obs_u")), fnum(r.get("obs_v"))],
                    "rm": [fnum(r.get("bunk_rm_u")), fnum(r.get("bunk_rm_v"))],
                    "prof": prof,
                })
        env_out = Path(a.out).with_name("env.json")
        env_out.write_text(json.dumps(env, separators=(",", ":")).replace("NaN", "null"))
        print(f"HRRR environments: {len(env)} meso-hours -> {env_out} ({env_out.stat().st_size/1e6:.1f} MB)")

    for s in supercells:
        ts = [p[0] for mi in s["mesos"] for p in mesos[mi]["pts"]]
        s["t0"], s["t1"] = min(ts), max(ts)
        s["id"] = f"{cases[s['case']]['date']} {s['label']}"

    # ---- tornadoes: whole tornadoes from the DB (oneTorID); NCEI download only as fallback ----
    def evid(ev):
        return int(float(ev)) if re.fullmatch(r"\d+(\.0)?", ev or "") else None

    resolved = {}  # (mi, evid) -> tornado dict or None
    need_ncei = set()
    for mi, ev, _ in pending_links:
        e = evid(ev)
        t = db.tornado(e) if (db is not None and e is not None) else None
        resolved[(mi, e)] = t
        if t is None and e is not None:
            need_ncei.add((mi, e))

    if need_ncei:
        years = {int(cases[supercells[mesos[mi]["sc"]]["case"]]["date"][:4]) for mi, _ in need_ncei}
        nc = ncei.load_tornadoes(years, Path(a.ncei_cache), Path(a.ncei_local) if a.ncei_local else None)
        by_id = {int(r.EVENT_ID): r for r in nc.itertuples()} if len(nc) else {}
        for mi, e in need_ncei:
            r = by_id.get(e)
            if r is None:
                continue
            lat0, lon0 = r.BEGIN_LAT, r.BEGIN_LON
            lat1 = r.END_LAT if not pd.isna(r.END_LAT) else lat0
            lon1 = r.END_LON if not pd.isna(r.END_LON) else lon0
            path = [[round(float(lat0), 4), round(float(lon0), 4)], [round(float(lat1), 4), round(float(lon1), 4)]]
            if any(math.isnan(v) for p in path for v in p):
                path = None
            ef = str(r.TOR_F_SCALE).strip().upper().replace("EF", "").replace("F", "") if not pd.isna(r.TOR_F_SCALE) else ""
            ef = ef if ef in {"0", "1", "2", "3", "4", "5"} else "U"
            seg = {"event": e, "t0": int(r.t0), "t1": int(r.t1), "path": path, "ef": ef,
                   "where": f"{str(r.CZ_NAME).title()}, {str(r.STATE).title()}",
                   "len": None if pd.isna(r.TOR_LENGTH) else round(float(r.TOR_LENGTH), 2),
                   "wid": None if pd.isna(r.TOR_WIDTH) else int(r.TOR_WIDTH)}
            resolved[(mi, e)] = {"id": e, "src": "ncei", "ncei": True, "t0": seg["t0"], "t1": seg["t1"], "ef": ef,
                                 "len": seg["len"], "wid": seg["wid"], "inj": int(r.INJURIES_DIRECT or 0),
                                 "dth": int(r.DEATHS_DIRECT or 0), "where": seg["where"], "segs": [seg],
                                 "partial": True}

    # one entry per whole tornado; a tornado linked to several mesos is shared by them
    tor_index = {}
    missing = 0
    for mi, ev, fname in pending_links:
        e = evid(ev)
        t = resolved.get((mi, e))
        if t is None:
            key = ("pending", e if e is not None else ev, fname)
            t = {"id": e, "src": None, "ncei": False, "segs": []}
        else:
            key = ("tor", t["id"])
        if key not in tor_index:
            tor_index[key] = len(tornadoes)
            t = dict(t, mesos=[], linked=[])
            tornadoes.append(t)
            if not t["ncei"]:
                missing += 1
        ti = tor_index[key]
        tor = tornadoes[ti]
        if e is not None and e not in tor["linked"]:
            tor["linked"].append(e)
        if mi not in tor["mesos"]:
            tor["mesos"].append(mi)
        if ti not in mesos[mi]["tor"]:
            mesos[mi]["tor"].append(ti)
    dupes = len(pending_links) - len(tornadoes)
    if dupes:
        print(f"Merged {dupes} tornado link(s) that were segments of an already-linked tornado")

    out = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": cases, "supercells": supercells, "mesos": mesos, "tornadoes": tornadoes,
        "ncei_missing": missing, "problems": problems,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, separators=(",", ":")))
    for p in problems:
        print("!", p)
    print(f"{len(cases)} days, {len(supercells)} supercells, {len(mesos)} mesocyclones, "
          f"{len(tornadoes)} whole tornadoes from {len(pending_links)} links ({missing} not found) -> {a.out}")


if __name__ == "__main__":
    main()
