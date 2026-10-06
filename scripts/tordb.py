"""Thea tornado database (1950–2024): one row per NCEI segment, grouped into whole
tornadoes by `oneTorID`. Times (`beginUnix`/`endUnix`) are UTC epochs; -99900 = missing.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

MISSING = -99900
COLS = ["stormEventsReportID", "oneTorID", "previousSegment", "beginUnix", "endUnix",
        "beginLatitude", "beginLongitude", "endLatitude", "endLongitude", "magnitude",
        "tornadoLength", "tornadoWidth", "injuriesDirect", "deathsDirect", "county", "state"]


def is_tordb(path: Path) -> bool:
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        header = fh.readline()
    return "oneTorID" in header and "stormEventsReportID" in header


def _num(v):
    return None if pd.isna(v) or v == MISSING else v


def _ef(v):
    s = str(v).strip().upper()
    return s if s in {"0", "1", "2", "3", "4", "5"} else "U"


class TorDB:
    def __init__(self, paths: list[Path]):
        frames = [pd.read_csv(p, usecols=lambda c: c in COLS, low_memory=False) for p in paths]
        df = pd.concat(frames, ignore_index=True).drop_duplicates("stormEventsReportID")
        df["stormEventsReportID"] = df["stormEventsReportID"].astype("int64")
        df["oneTorID"] = df["oneTorID"].astype("int64")
        self.df = df
        self.by_event = dict(zip(df["stormEventsReportID"], df["oneTorID"]))
        self.groups = {k: g.sort_values("beginUnix") for k, g in df.groupby("oneTorID")}
        print(f"Tornado DB: {len(df)} segments, {len(self.groups)} tornadoes "
              f"({int(df.beginUnix.min())}–{int(df.beginUnix.max())})")

    def tornado(self, event_id: int) -> dict | None:
        """Whole-tornado record for any segment's NCEI event id, or None if not in the DB."""
        tid = self.by_event.get(event_id)
        if tid is None:
            return None
        g = self.groups[tid]
        segs = []
        for r in g.itertuples():
            lat0, lon0 = _num(r.beginLatitude), _num(r.beginLongitude)
            lat1, lon1 = _num(r.endLatitude), _num(r.endLongitude)
            path = None
            if lat0 is not None and lon0 is not None:
                if lat1 is None or lon1 is None:
                    lat1, lon1 = lat0, lon0
                path = [[round(float(lat0), 4), round(float(lon0), 4)], [round(float(lat1), 4), round(float(lon1), 4)]]
            t1 = _num(r.endUnix)
            segs.append({
                "event": int(r.stormEventsReportID),
                "t0": int(r.beginUnix), "t1": int(t1) if t1 is not None else int(r.beginUnix),
                "path": path, "ef": _ef(r.magnitude),
                "where": f"{str(r.county).title()}, {str(r.state).title()}",
                "len": None if _num(r.tornadoLength) is None else round(float(r.tornadoLength), 2),
                "wid": None if _num(r.tornadoWidth) is None else int(r.tornadoWidth),
            })
        efs = [int(s["ef"]) for s in segs if s["ef"] != "U"]
        lens = [s["len"] for s in segs if s["len"] is not None]
        wids = [s["wid"] for s in segs if s["wid"] is not None]
        inj = g["injuriesDirect"].where(g["injuriesDirect"] != MISSING, 0).fillna(0).sum()
        dth = g["deathsDirect"].where(g["deathsDirect"] != MISSING, 0).fillna(0).sum()
        states = list(dict.fromkeys(str(s).title() for s in g["state"]))
        counties = list(dict.fromkeys(str(c).title() for c in g["county"]))
        return {
            "id": int(tid), "src": "db", "ncei": True,
            "t0": min(s["t0"] for s in segs), "t1": max(s["t1"] for s in segs),
            "ef": str(max(efs)) if efs else "U",
            "len": round(sum(lens), 2) if lens else None,
            "wid": max(wids) if wids else None,
            "inj": int(inj), "dth": int(dth),
            "where": (", ".join(counties[:3]) + (f" +{len(counties) - 3}" if len(counties) > 3 else "")) + ", " + "/".join(states),
            "segs": segs,
        }
