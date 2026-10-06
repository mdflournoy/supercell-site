"""Fetch and cache NCEI Storm Events tornado records (details files), by year.

Files live at https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/
named StormEvents_details-ftp_v1.0_dYYYY_cYYYYMMDD.csv.gz; we pick the newest
revision for each year we need and cache it on disk.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
PAT = re.compile(r"StormEvents_details-ftp_v1\.0_d(\d{4})_c(\d{8})\.csv\.gz")
TZ_HOURS = {"EST": 5, "CST": 6, "MST": 7, "PST": 8, "AKST": 9, "HST": 10, "AST": 4, "SST": 11, "GST": -10}
KEEP = ["EVENT_ID", "EVENT_TYPE", "STATE", "CZ_NAME", "WFO", "BEGIN_YEARMONTH", "BEGIN_DAY", "BEGIN_TIME",
        "END_YEARMONTH", "END_DAY", "END_TIME", "CZ_TIMEZONE", "TOR_F_SCALE", "TOR_LENGTH", "TOR_WIDTH",
        "BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON", "INJURIES_DIRECT", "DEATHS_DIRECT"]


def _utc_offset_hours(tz: str) -> float:
    m = re.match(r"([A-Za-z]+)-?(\d+)?", str(tz).strip())
    if not m:
        return 6.0
    if m.group(2):
        return float(m.group(2))
    return float(TZ_HOURS.get(m.group(1).upper(), 6))


def _epoch(ym, day, hhmm, tz) -> float:
    ym, day, hhmm = int(ym), int(day), int(hhmm)
    ts = pd.Timestamp(year=ym // 100, month=ym % 100, day=day, hour=hhmm // 100, minute=hhmm % 100, tz="UTC")
    ts = ts + pd.Timedelta(hours=_utc_offset_hours(tz))  # local standard time -> UTC
    return ts.timestamp()


def latest_files(years: set[int]) -> dict[int, str]:
    html = requests.get(BASE, timeout=60).text
    best: dict[int, tuple[str, str]] = {}
    for name, y, c in {(m.group(0), int(m.group(1)), m.group(2)) for m in PAT.finditer(html)}:
        if y in years and (y not in best or c > best[y][1]):
            best[y] = (name, c)
    return {y: v[0] for y, v in best.items()}


def load_tornadoes(years: set[int], cache: Path, local_dir: Path | None = None) -> pd.DataFrame:
    """Return tornado rows for the given years with UTC epoch begin/end times."""
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    if local_dir and local_dir.exists():  # offline/testing: pre-placed csv(.gz) files
        paths = sorted(local_dir.glob("StormEvents_details*.csv*"))
    else:
        try:
            names = latest_files(years)
        except Exception as e:  # noqa: BLE001
            print(f"! Could not list NCEI directory ({e}); using cache only")
            names = {}
        paths = []
        for y in sorted(years):
            name = names.get(y)
            if name is None:
                cached = sorted(cache.glob(f"StormEvents_details-ftp_v1.0_d{y}_c*.csv.gz"))
                if cached:
                    paths.append(cached[-1])
                else:
                    print(f"! No NCEI details file for {y} yet")
                continue
            p = cache / name
            if not p.exists():
                print(f"  downloading {name}")
                for old in cache.glob(f"StormEvents_details-ftp_v1.0_d{y}_c*.csv.gz"):
                    old.unlink()
                r = requests.get(BASE + name, timeout=300)
                r.raise_for_status()
                p.write_bytes(r.content)
            paths.append(p)
    for p in paths:
        df = pd.read_csv(p, usecols=lambda c: c in KEEP, low_memory=False)
        frames.append(df[df["EVENT_TYPE"].str.strip().str.lower() == "tornado"])
    if not frames:
        return pd.DataFrame(columns=KEEP + ["t0", "t1"])
    df = pd.concat(frames, ignore_index=True)
    df["t0"] = [_epoch(a, b, c, z) for a, b, c, z in zip(df.BEGIN_YEARMONTH, df.BEGIN_DAY, df.BEGIN_TIME, df.CZ_TIMEZONE)]
    df["t1"] = [_epoch(a, b, c, z) for a, b, c, z in zip(df.END_YEARMONTH, df.END_DAY, df.END_TIME, df.CZ_TIMEZONE)]
    return df
