#!/usr/bin/env python3
"""Write renumbered copies of the MesoTrack day files.

For each day file in --raw (the originals from Google Drive, left untouched):
  * supercells are relettered A, B, C, ... Z, AA, AB, ... in order of their start
    time (earliest scan of any of their mesocyclones);
  * mesocyclones within each supercell are renumbered 1, 2, 3, ... in order of
    their own start time;
  * the "Tornadoes" link keys (e.g. "C4") and the primary_sup merge column are
    remapped to the new IDs.

Output (in --out, which is cleared of old CSVs first):
  YYYYMMDD.csv   same layout as the original, new IDs
  id_map.csv     date, file, old/new supercell and meso IDs for every mesocyclone

Usage: python scripts/renumber.py --raw data/raw --out tracks
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import tordb  # noqa: E402

LINK = re.compile(r"^\s*([A-Za-z]+)\s*0*(\d+)\s*$")


def letters(i: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, 27 -> AB, ..."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def natural(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def clean_id(v) -> str:
    s = str(v).strip()
    return s[:-2] if re.fullmatch(r"\d+\.0", s) else s


def renumber_file(path: Path, out_dir: Path) -> tuple[list[dict], list[str]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    split = next((i for i, ln in enumerate(lines) if ln.strip().strip(",").lower() == "tornadoes"), len(lines))
    track_lines = [ln for ln in lines[:split] if ln.strip().strip(",")]
    link_lines = [ln for ln in lines[split + 1:] if ln.strip().strip(",")]
    problems = []

    df = pd.read_csv(io.StringIO("\n".join(track_lines)), dtype=str, keep_default_na=False)
    cols = {c.strip().lower(): c for c in df.columns}
    need = ["supercell_id", "meso_id", "time_utc"]
    if any(n not in cols for n in need):
        return [], [f"{path.name}: missing columns {[n for n in need if n not in cols]}; not renumbered"]
    SC, MS, TM = cols["supercell_id"], cols["meso_id"], cols["time_utc"]
    df[SC] = df[SC].map(clean_id)
    df[MS] = df[MS].map(clean_id)
    t = pd.to_datetime(df[TM], utc=True, errors="coerce", format="mixed")

    # supercell order: earliest scan; ties broken by original ID
    sc_start = t.groupby(df[SC]).min()
    sc_order = sorted(sc_start.index, key=lambda s: (sc_start[s] if pd.notna(sc_start[s]) else pd.Timestamp.max.tz_localize("UTC"), natural(s)))
    sc_map = {old: letters(i) for i, old in enumerate(sc_order)}

    # meso order within each supercell: earliest scan; ties broken by original ID
    meso_map = {}
    for sc in sc_order:
        sub = df[SC] == sc
        ms_start = t[sub].groupby(df.loc[sub, MS]).min()
        order = sorted(ms_start.index, key=lambda m: (ms_start[m] if pd.notna(ms_start[m]) else pd.Timestamp.max.tz_localize("UTC"), natural(m)))
        for j, m in enumerate(order):
            meso_map[(sc, m)] = str(j + 1)

    new = df.copy()
    new[SC] = df[SC].map(sc_map)
    new[MS] = [meso_map[(s, m)] for s, m in zip(df[SC], df[MS])]
    if "primary_sup" in cols:
        P = cols["primary_sup"]
        def remap_primary(v):
            v = clean_id(v)
            if not v:
                return v
            if v in sc_map:
                return sc_map[v]
            problems.append(f"{path.name}: primary_sup '{v}' isn't a supercell in this file; left unchanged")
            return v
        new[P] = df[P].map(remap_primary)

    # keep rows grouped by new supercell / meso, in time order
    order_key = pd.DataFrame({"s": new[SC].map(lambda s: (len(s), s)), "m": new[MS].astype(int), "t": t})
    new = new.loc[order_key.sort_values(["s", "m", "t"], kind="stable").index]

    # tornado links: "<supercell><meso>,<event id>"
    key_map = {f"{s}{m}".upper(): f"{sc_map[s]}{meso_map[(s, m)]}" for (s, m) in meso_map}
    new_links = []
    for ln in link_lines:
        parts = ln.split(",")
        key = parts[0].strip()
        mk = LINK.match(key)
        norm = f"{mk.group(1)}{mk.group(2)}".upper() if mk else key.upper()
        if norm in key_map:
            parts[0] = key_map[norm]
        else:
            problems.append(f"{path.name}: tornado link '{key}' doesn't match any meso; left unchanged")
        new_links.append(",".join(p.strip() for p in parts))

    out = new.to_csv(index=False, lineterminator="\n")
    if link_lines or split < len(lines):
        out += "\nTornadoes\n" + "".join(l + "\n" for l in new_links)
    (out_dir / path.name).write_text(out, encoding="utf-8")

    m = re.search(r"(20\d{6})", path.stem)
    day = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}" if m else ""
    rows = [{"date": day, "file": path.name, "old_supercell": s, "old_meso": mm,
             "new_supercell": sc_map[s], "new_meso": meso_map[(s, mm)],
             "changed": "yes" if (sc_map[s], meso_map[(s, mm)]) != (s, mm) else ""}
            for (s, mm) in meso_map]
    rows.sort(key=lambda r: ((len(r["new_supercell"]), r["new_supercell"]), int(r["new_meso"])))
    return rows, problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="tracks")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.csv"):
        old.unlink()
    files = sorted(p for p in raw.rglob("*") if p.suffix.lower() == ".csv" and not tordb.is_tordb(p))
    all_rows, problems = [], []
    for p in files:
        try:
            rows, probs = renumber_file(p, out)
        except Exception as e:  # noqa: BLE001
            rows, probs = [], [f"{p.name}: couldn't renumber ({e})"]
        all_rows += rows
        problems += probs
    pd.DataFrame(all_rows, columns=["date", "file", "old_supercell", "old_meso", "new_supercell", "new_meso", "changed"]) \
        .to_csv(out / "id_map.csv", index=False, lineterminator="\n")
    for pr in problems:
        print("!", pr)
    changed = sum(1 for r in all_rows if r["changed"])
    print(f"Renumbered {len(files)} file(s): {len(all_rows)} mesocyclones, {changed} with new IDs -> {out}/")


if __name__ == "__main__":
    main()
