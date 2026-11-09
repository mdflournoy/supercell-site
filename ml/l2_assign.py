#!/usr/bin/env python3
"""Single-site stage 1: which NEXRAD did the analyst use for each hand-made track point?

Track times are radar volume start times (to the second), and every Level II file name carries its
volume start time, so a track point is matched to the radar(s) whose volume began within MATCH_S of it.
For every point we also record the nearest and second-nearest radar, to check the analysts' rule
("the closest radar, occasionally the second closest when the storm is very close to a site").

Writes ml/l2/radar_assign.csv (one row per track point), ml/l2/nexrad_sites.csv and prints a summary.
Needs network access to the NEXRAD Level II archive on AWS (run in GitHub Actions: ml-l2-assign.yml).
"""
from __future__ import annotations

import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ML_DIR, read_tracks  # noqa: E402

BUCKETS = ["https://unidata-nexrad-level2.s3.amazonaws.com", "https://noaa-nexrad-level2.s3.amazonaws.com"]
STATIONS = "https://www.ncei.noaa.gov/access/homr/file/nexrad-stations.txt"
S3NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
KEY_RE = re.compile(r"/([KPT][A-Z]{3})(\d{8})_(\d{6})(?:_V\d+)?(?:\.gz)?$")
MATCH_S = 10          # track time vs volume start
NEAR_KM = 300         # radars considered for a day


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


def sites() -> pd.DataFrame:
    rows = []
    for ln in get(STATIONS).text.splitlines():
        m = re.match(r"^\s*\d+\s+([KPT][A-Z]{3})\s.*?\s(-?\d{1,2}\.\d+)\s+(-?\d{2,3}\.\d+)\s+(-?\d+)\s.*NEXRAD", ln)
        if m:
            rows.append({"site": m.group(1), "lat": float(m.group(2)), "lon": float(m.group(3)),
                         "elev_ft": int(m.group(4))})
    return pd.DataFrame(rows).drop_duplicates("site")


def list_volumes(site: str, day: datetime) -> np.ndarray:
    """Volume start times (epoch s) of one radar on one UTC day."""
    prefix = f"{day:%Y/%m/%d}/{site}/"
    for bucket in BUCKETS:
        out, token = [], None
        try:
            while True:
                params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
                if token:
                    params["continuation-token"] = token
                root = ET.fromstring(get(bucket + "/", params=params).content)
                for c in root.iter(S3NS + "Contents"):
                    k = c.find(S3NS + "Key").text
                    if k.endswith("_MDM"):
                        continue
                    m = KEY_RE.search(k)
                    if m:
                        t = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
                        out.append(t.timestamp())
                if root.findtext(S3NS + "IsTruncated") == "true":
                    token = root.findtext(S3NS + "NextContinuationToken")
                else:
                    break
        except Exception as e:  # noqa: BLE001
            print("list failed", bucket, prefix, e)
            continue
        if out:
            return np.unique(out)
    return np.array([])


def km(lat0, lon0, lat1, lon1):
    p0, p1 = np.radians(lat0), np.radians(lat1)
    a = np.sin((p1 - p0) / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(np.radians(lon1 - lon0) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def main() -> None:
    out_dir = ML_DIR / "l2"
    out_dir.mkdir(exist_ok=True)
    st = sites()
    st.to_csv(out_dir / "nexrad_sites.csv", index=False)
    print(len(st), "NEXRAD sites")
    tr = read_tracks()
    rows = []
    for f, g in tr.groupby("file"):
        d = km(g.lat.to_numpy()[:, None], g.lon.to_numpy()[:, None], st.lat.to_numpy()[None], st.lon.to_numpy()[None])
        cand = np.where(d.min(axis=0) < NEAR_KM)[0]
        days = sorted({datetime.fromtimestamp(t, timezone.utc).date() for t in (g.t.min() - 600, g.t.max() + 600)})
        days = [days[0] + timedelta(n) for n in range((days[-1] - days[0]).days + 1)]
        vols = {}
        for j in cand:
            v = np.concatenate([list_volumes(st.site.iloc[j], datetime.combine(dd, datetime.min.time())) for dd in days])
            vols[j] = np.sort(v)
        print(f, len(g), "points;", len(cand), "radars;", sum(len(v) for v in vols.values()), "volumes", flush=True)
        order = np.argsort(d, axis=1)
        for i, (_, r) in enumerate(g.iterrows()):
            near = order[i]
            matches = []
            for j in cand:
                v = vols[j]
                if len(v):
                    k = np.searchsorted(v, r.t)
                    dt = min(abs(v[max(k - 1, 0)] - r.t), abs(v[min(k, len(v) - 1)] - r.t))
                    if dt <= MATCH_S:
                        matches.append((dt, d[i, j], j))
            matches.sort()
            rank = {j: n + 1 for n, j in enumerate(near[:10])}
            best = matches[0] if matches else None
            rows.append({
                "file": f, "label": r.label, "t": r.t, "lat": r.lat, "lon": r.lon,
                "near1": st.site.iloc[near[0]], "d1_km": round(d[i, near[0]], 1),
                "near2": st.site.iloc[near[1]], "d2_km": round(d[i, near[1]], 1),
                "radar": st.site.iloc[best[2]] if best else "", "radar_km": round(best[1], 1) if best else np.nan,
                "radar_rank": rank.get(best[2], 99) if best else 0, "dt_s": best[0] if best else np.nan,
                "n_match": len(matches),
                "matches": ";".join(f"{st.site.iloc[j]}:{int(dt)}" for dt, _, j in matches),
            })
    a = pd.DataFrame(rows)
    a.to_csv(out_dir / "radar_assign.csv", index=False)
    m = a[a.n_match > 0]
    print(f"\n{len(a)} track points; {len(m)} ({len(m) / len(a):.1%}) match a volume start within {MATCH_S} s")
    print(f"ambiguous (2+ radars match): {(m.n_match > 1).mean():.1%}")
    u = m[m.n_match == 1]
    print("rank of the matched radar (1 = nearest), unambiguous points:")
    print(u.radar_rank.value_counts(normalize=True).sort_index().round(3).to_string())
    print("distance to nearest radar when the 2nd nearest was used (km):")
    print(u[u.radar_rank == 2].d1_km.describe().round(1).to_string())
    print("distance to nearest radar when the nearest was used (km):")
    print(u[u.radar_rank == 1].d1_km.describe().round(1).to_string())
    print("by day: share matched, share nearest")
    print(a.groupby("file").apply(lambda x: pd.Series({"n": len(x), "matched": (x.n_match > 0).mean(),
                                                       "nearest": (x.radar_rank == 1).mean()})).round(2).to_string())


if __name__ == "__main__":
    main()
