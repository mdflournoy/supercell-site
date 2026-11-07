"""Shared helpers for the mesocyclone-tracking model (ml/).

Domain grid
    The Southeast analysis box on the MRMS 0.005° AzShear grid (lat 28–40.5, lon -97.5 to -77.5).
    Reflectivity (0.01°) is stored on its own grid over the same box, half the resolution.

Labels
    Track files in tracks/ (renumbered IDs). Each mesocyclone is linearly interpolated to a frame
    time only within its tracked lifetime (first to last scan); a single-scan meso counts at frames
    within ±2 min of its scan.

Southeast region
    AR, LA, MS, AL, GA, TN, KY (ml/static/se_states.geojson). Every supercell whose track entered these
    states was tracked, so inside them anything untracked is "no mesocyclone". Outside them the
    truth is unknown except near tracks.
"""
from __future__ import annotations

import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent
REPO = ML_DIR.parent

# ---------------------------------------------------------------------------- grid
LAT0, LAT1 = 28.0, 40.5        # box edges (degrees)
LON0, LON1 = -97.5, -77.5
DX_AZ = 0.005                  # MRMS AzShear grid spacing
DX_REF = 0.01                  # MRMS reflectivity grid spacing


def box_shape(dx: float) -> tuple[int, int]:
    return int(round((LAT1 - LAT0) / dx)), int(round((LON1 - LON0) / dx))


def box_centers(dx: float):
    """Cell-center latitudes (north to south, like MRMS) and longitudes (west to east) of the box."""
    ny, nx = box_shape(dx)
    lats = LAT1 - dx / 2 - np.arange(ny) * dx
    lons = LON0 + dx / 2 + np.arange(nx) * dx
    return lats, lons


def latlon_to_rc(lat, lon, dx: float = DX_AZ):
    """Fractional (row, col) in the box grid for lat/lon (row 0 = north edge)."""
    return (LAT1 - np.asarray(lat)) / dx - 0.5, (np.asarray(lon) - LON0) / dx - 0.5


# ---------------------------------------------------------------------------- radars
def radar_sites() -> pd.DataFrame:
    return pd.read_csv(ML_DIR / "static" / "nexrad_sites.csv")


def radar_fields(dx: float = DX_AZ, max_range_km: float = 460.0):
    """(distance to nearest WSR-88D in km, lowest 0.5° beam-center height in m above that radar)
    on the box grid; 4/3-earth beam model; the beam height is the minimum over radars in range."""
    lats, lons = box_centers(dx)
    r = radar_sites()
    r = r[(r.lat > LAT0 - 5) & (r.lat < LAT1 + 5) & (r.lon > LON0 - 6) & (r.lon < LON1 + 6)]
    ke_a = 4.0 / 3.0 * 6371e3
    el = math.radians(0.5)
    dist = np.full((len(lats), len(lons)), np.inf, np.float32)
    beam = np.full((len(lats), len(lons)), np.inf, np.float32)
    LO = lons[None, :]
    for _, s in r.iterrows():
        for j0 in range(0, len(lats), 500):  # chunk rows to bound memory
            LA = lats[j0:j0 + 500, None]
            p1, p2 = math.radians(s.lat), np.radians(LA)
            h = np.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * np.cos(p2) * np.sin(np.radians(LO - s.lon) / 2) ** 2
            d = 12742.0 * np.arcsin(np.sqrt(h))  # km
            rng = d * 1e3
            hb = np.sqrt(rng ** 2 + ke_a ** 2 + 2 * rng * ke_a * math.sin(el)) - ke_a
            hb = np.where(d <= max_range_km, hb, np.inf)
            np.minimum(dist[j0:j0 + 500], d, out=dist[j0:j0 + 500])
            np.minimum(beam[j0:j0 + 500], hb, out=beam[j0:j0 + 500])
    return dist, beam


def nearest_radar(lat, lon):
    """Distance (km) to and lowest beam height (m) from the nearest WSR-88D for points."""
    r = radar_sites()
    lat, lon = np.asarray(lat, float)[:, None], np.asarray(lon, float)[:, None]
    p1, p2 = np.radians(r.lat.to_numpy())[None, :], np.radians(lat)
    h = np.sin((p1 - p2) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(r.lon.to_numpy()[None, :] - lon) / 2) ** 2
    d = 12742.0 * np.arcsin(np.sqrt(h))
    k = d.argmin(1)
    dmin = d[np.arange(len(d)), k]
    ke_a, el = 4.0 / 3.0 * 6371e3, math.radians(0.5)
    rng = dmin * 1e3
    hb = np.sqrt(rng ** 2 + ke_a ** 2 + 2 * rng * ke_a * math.sin(el)) - ke_a
    return dmin, hb, r.site.to_numpy()[k]


# ---------------------------------------------------------------------------- Southeast mask
def se_polygons():
    g = json.loads((ML_DIR / "static" / "se_states.geojson").read_text())
    polys = []
    for f in g["features"]:
        geom = f["geometry"]
        parts = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for part in parts:
            polys.append(np.asarray(part[0]))  # outer ring
    return polys


def in_southeast(lat, lon) -> np.ndarray:
    from matplotlib.path import Path as MPath

    pts = np.column_stack([np.ravel(lon), np.ravel(lat)])
    inside = np.zeros(len(pts), bool)
    for ring in se_polygons():
        inside |= MPath(ring).contains_points(pts)
    return inside.reshape(np.shape(lat))


def se_mask(dx: float = DX_AZ) -> np.ndarray:
    lats, lons = box_centers(dx)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    return in_southeast(LA, LO)


# ---------------------------------------------------------------------------- tracks
def read_tracks(tracks_dir: Path | str = REPO / "tracks") -> pd.DataFrame:
    """All track points: file (YYYYMMDD), supercell, meso, label (e.g. A1), t (epoch s), lat, lon,
    tornadic (meso linked to a tornado), sc_tornadic (any meso of its supercell linked)."""
    rows = []
    for p in sorted(Path(tracks_dir).glob("2*.csv")):
        lines = p.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n")
        k = next((i for i, ln in enumerate(lines) if ln.strip().strip(",").lower() == "tornadoes"), len(lines))
        df = pd.read_csv(io.StringIO("\n".join(ln for ln in lines[:k] if ln.strip().strip(","))), dtype=str)
        tor = set()
        for ln in lines[k + 1:]:
            parts = [x.strip() for x in ln.split(",")]
            if parts and parts[0]:
                tor.add(parts[0].upper())
        t = pd.to_datetime(df["time_utc"], utc=True, format="mixed")
        out = pd.DataFrame({
            "file": p.stem, "supercell": df["supercell_id"].str.strip(), "meso": df["meso_id"].str.strip(),
            "t": (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds(),
            "lat": pd.to_numeric(df["lat"]), "lon": pd.to_numeric(df["lon"]),
        })
        out["label"] = out["supercell"] + out["meso"]
        out["tornadic"] = out["label"].str.upper().isin(tor)
        sc_tor = {lab.rstrip("0123456789") for lab in tor}
        out["sc_tornadic"] = out["supercell"].str.upper().isin(sc_tor)
        rows.append(out)
    tr = pd.concat(rows, ignore_index=True).sort_values(["file", "supercell", "meso", "t"])
    return tr.reset_index(drop=True)


def meso_positions_at(tr: pd.DataFrame, t: float, single_scan_tol: float = 120.0) -> pd.DataFrame:
    """Every meso active at time t (epoch s), linearly interpolated (no extrapolation)."""
    out = []
    for (f, sc, ms), g in tr.groupby(["file", "supercell", "meso"], sort=False):
        ts = g["t"].to_numpy()
        if len(ts) == 1:
            if abs(ts[0] - t) > single_scan_tol:
                continue
            la, lo = g["lat"].iloc[0], g["lon"].iloc[0]
        else:
            if t < ts[0] or t > ts[-1]:
                continue
            la = float(np.interp(t, ts, g["lat"].to_numpy()))
            lo = float(np.interp(t, ts, g["lon"].to_numpy()))
        r0 = g.iloc[0]
        out.append((f, sc, ms, r0["label"], la, lo, bool(r0["tornadic"]), bool(r0["sc_tornadic"])))
    return pd.DataFrame(out, columns=["file", "supercell", "meso", "label", "lat", "lon", "tornadic", "sc_tornadic"])


def analysis_windows(tr: pd.DataFrame, pad_s: float = 3600.0) -> dict[str, tuple[float, float]]:
    """Per track file, the window used for frames: first scan - 1 h to last scan + 1 h."""
    return {f: (g["t"].min() - pad_s, g["t"].max() + pad_s) for f, g in tr.groupby("file")}


def utc(t: float) -> datetime:
    return datetime.fromtimestamp(t, timezone.utc)
