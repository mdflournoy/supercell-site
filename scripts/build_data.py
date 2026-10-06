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
    a = ap.parse_args()

    files = sorted(p for p in Path(a.raw).rglob("*") if p.suffix.lower() == ".csv")
    if not files:
        sys.exit(f"No CSVs in {a.raw}")

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
                supercells.append({"case": case_idx, "label": sc, "mesos": [], "merged_into": None})
            si = sc_index[sc]
            mi = len(mesos)
            end = str(g["dissipation"].dropna().iloc[-1]).strip() if "dissipation" in g and g["dissipation"].notna().any() else ""
            prim = str(g["primary_sup"].dropna().iloc[-1]).strip() if "primary_sup" in g and g["primary_sup"].notna().any() else ""
            if prim:
                supercells[si]["merged_into"] = prim
            mesos.append({
                "sc": si, "label": f"{sc}{mid}", "end": end,
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

    for s in supercells:
        ts = [p[0] for mi in s["mesos"] for p in mesos[mi]["pts"]]
        s["t0"], s["t1"] = min(ts), max(ts)
        s["id"] = f"{cases[s['case']]['date']} {s['label']}"

    # ---- NCEI tornado records ------------------------------------------------------------
    years = {int(cases[supercells[mesos[mi]["sc"]]["case"]]["date"][:4]) for mi, _, _ in pending_links}
    nc = ncei.load_tornadoes(years, Path(a.ncei_cache), Path(a.ncei_local) if a.ncei_local else None) if years else pd.DataFrame()
    by_id = {int(r.EVENT_ID): r for r in nc.itertuples()} if len(nc) else {}

    missing = 0
    for mi, ev, fname in pending_links:
        evid = int(float(ev)) if re.fullmatch(r"\d+(\.0)?", ev or "") else None
        r = by_id.get(evid) if evid is not None else None
        t = {"event": evid, "meso": mi, "ncei": r is not None}
        if r is not None:
            lat0, lon0 = r.BEGIN_LAT, r.BEGIN_LON
            lat1 = r.END_LAT if not pd.isna(r.END_LAT) else lat0
            lon1 = r.END_LON if not pd.isna(r.END_LON) else lon0
            ef = str(r.TOR_F_SCALE).strip().upper() if not pd.isna(r.TOR_F_SCALE) else ""
            t.update({
                "t0": int(r.t0), "t1": int(r.t1),
                "path": [[round(float(lat0), 4), round(float(lon0), 4)], [round(float(lat1), 4), round(float(lon1), 4)]],
                "ef": ef.replace("EF", "").replace("F", "") or "U",
                "where": f"{str(r.CZ_NAME).title()}, {str(r.STATE).title()}",
                "len": None if pd.isna(r.TOR_LENGTH) else round(float(r.TOR_LENGTH), 2),
                "wid": None if pd.isna(r.TOR_WIDTH) else int(r.TOR_WIDTH),
                "inj": int(r.INJURIES_DIRECT or 0), "dth": int(r.DEATHS_DIRECT or 0),
            })
            if any(math.isnan(v) for p in t["path"] for v in p):
                t["path"] = None
        else:
            missing += 1
        mesos[mi]["tor"].append(len(tornadoes))
        tornadoes.append(t)

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
          f"{len(tornadoes)} tornadoes ({missing} not yet in NCEI) -> {a.out}")


if __name__ == "__main__":
    main()
