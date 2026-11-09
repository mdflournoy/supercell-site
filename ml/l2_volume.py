#!/usr/bin/env python3
"""Single-site stage 2: one NEXRAD Level II volume -> model input on a radar-centered grid.

Steps for a volume:
  1. Read with Py-ART. Split-cut / SAILS volumes repeat elevations: for each elevation the first
     surveillance sweep supplies reflectivity (Z), ZDR and CC; the first Doppler sweep supplies
     velocity (V) and spectrum width (SW).
  2. Dealias V with the MIT Lincoln Laboratory U-Net velocity dealiaser (Veillette et al. 2023,
     AIES; github.com/mit-ll/unet-vda), the algorithm the analysts use. 0.5-degree sweeps are split
     into two 1-degree sweeps as in the reference code.
  3. From dealiased V, local linear least squares (LLSD-style) azimuthal shear dV/ds and radial
     divergence dV/dr on each sweep (s^-1).
  4. Height layers: for every grid cell, each field is taken from the sweep whose beam center
     (4/3-earth model) is closest to the layer height above the radar (within LAYER_TOL_KM), so a
     meso is seen at about the same height at any range. "low" is always the lowest sweep.
  5. Nearest-gate sampling onto a GRID_KM x GRID_KM grid at DX_KM centered on the radar
     (azimuthal equidistant), normalized.

Channels (CHANNELS): Z, AzShear, Div, SW at low/1.5/3/4.5/6 km; ZDR, CC at low/1.5/3 km; range;
height of the lowest beam.
"""
from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # unet-vda is Keras 2 code (needs the tf_keras package)

DX_KM = 0.5
GRID_KM = 400.0                    # grid is +/- 200 km
N = int(GRID_KM / DX_KM)           # 800
MAX_RANGE_KM = 200.0
KE_A = 4.0 / 3.0 * 6371.0          # effective earth radius (km)
LAYERS = ["low", 1.5, 3.0, 4.5, 6.0]
DP_LAYERS = ["low", 1.5, 3.0]
LAYER_TOL_KM = 1.0
VDA_GATES = 896                    # 224 km of 250 m gates
VDA_PAD = 12

CHANNELS = ([f"{f}_{h}" for h in LAYERS for f in ("Z", "AzSh", "Div", "SW")]
            + [f"{f}_{h}" for h in DP_LAYERS for f in ("ZDR", "CC")] + ["range", "beam_low"])
NCH = len(CHANNELS)


# ---------------------------------------------------------------- geometry
def grid_xy():
    c = (np.arange(N) - N / 2 + 0.5) * DX_KM
    x, y = np.meshgrid(c, -c)                          # row 0 = north
    return x, y


@lru_cache(maxsize=1)
def grid_polar():
    x, y = grid_xy()
    s = np.hypot(x, y)
    az = np.degrees(np.arctan2(x, y)) % 360
    return s.astype(np.float32), az.astype(np.float32)


def slant_and_height(s_km: np.ndarray, elev_deg: float):
    """Ground distance -> (slant range, beam-center height above radar), 4/3-earth model."""
    e = np.radians(elev_deg)
    th = s_km / KE_A
    # geometry: radar at distance KE_A from center; beam point at angle th; solve for r with elevation e
    r = KE_A * np.sin(th) / np.cos(e + th)
    h = np.sqrt(r ** 2 + KE_A ** 2 + 2 * r * KE_A * np.sin(e)) - KE_A
    return r, h


def aeqd_to_latlon(x_km, y_km, lat0, lon0):
    import pyart
    lon, lat = pyart.core.cartesian_to_geographic_aeqd(np.asarray(x_km) * 1000, np.asarray(y_km) * 1000, lon0, lat0)
    return lat, lon


def latlon_to_rc(lat, lon, lat0, lon0):
    import pyart
    x, y = pyart.core.geographic_to_cartesian_aeqd(np.asarray(lon, float), np.asarray(lat, float), lon0, lat0)
    col = x / 1000 / DX_KM + N / 2 - 0.5
    row = -y / 1000 / DX_KM + N / 2 - 0.5
    return row, col


def rc_to_latlon(row, col, lat0, lon0):
    x = (np.asarray(col) - N / 2 + 0.5) * DX_KM
    y = -(np.asarray(row) - N / 2 + 0.5) * DX_KM
    return aeqd_to_latlon(x, y, lat0, lon0)


# ---------------------------------------------------------------- dealiasing
_VDA = None


def vda_model():
    global _VDA
    if _VDA is None:
        root = Path(os.environ.get("UNET_VDA", Path(__file__).resolve().parent / "unet-vda"))
        sys.path.insert(0, str(root / "src"))
        import tensorflow as tf  # noqa: F401
        from dealias import VelocityDealiaser
        from feature_extraction import create_downsampler, create_upsampler
        import tensorflow as tf
        vel = tf.keras.Input(shape=(None, None, 1))
        down = create_downsampler(inp=vel, start_neurons=16, input_channels=1)
        up = create_upsampler(n_inputs=1, start_neurons=16, n_outputs=6)
        m = VelocityDealiaser(down, up)
        m.load_weights(str(root / "models" / "dealias_sn16_csi9764.SavedModel"))
        _VDA = m
    return _VDA


def dealias(sweeps: list[tuple[np.ndarray, float]]) -> list[np.ndarray]:
    """sweeps: (velocity [naz sorted by azimuth, ngates] with NaN for missing, nyquist). Returns dealiased."""
    if not sweeps:
        return []
    batch, nyq, back = [], [], []
    for k, (v, ny) in enumerate(sweeps):
        g = v.shape[1]
        a = np.full((v.shape[0], VDA_GATES), np.nan, np.float32)
        a[:, :min(g, VDA_GATES)] = v[:, :VDA_GATES]
        parts = [a[0::2], a[1::2]] if v.shape[0] == 720 else [a]
        for j, p in enumerate(parts):
            if p.shape[0] != 360:  # unusual radial count: resample to 360 nearest
                p = p[np.linspace(0, p.shape[0] - 1, 360).round().astype(int)]
            p = np.concatenate([p[-VDA_PAD:], p, p[:VDA_PAD]], axis=0)
            batch.append(p); nyq.append(ny); back.append((k, j, len(parts)))
    X = np.stack(batch)[:, None, :, :, None]
    Ny = np.array(nyq, np.float32)[:, None, None]
    out = vda_model().predict({"vel": X, "nyq": Ny}, batch_size=4, verbose=0)["dealiased_vel"][:, VDA_PAD:-VDA_PAD, :, 0]
    res = [np.full_like(v, np.nan) for v, _ in sweeps]
    for o, (k, j, n) in zip(out, back):
        v = sweeps[k][0]
        g = min(v.shape[1], VDA_GATES)
        if n == 2:
            res[k][j::2, :g] = o[:, :g]
        elif v.shape[0] == 360:
            res[k][:, :g] = o[:, :g]
        else:
            idx = np.linspace(0, v.shape[0] - 1, 360).round().astype(int)
            res[k][idx, :g] = o[:, :g]
    for k, (v, _) in enumerate(sweeps):
        res[k][np.isnan(v)] = np.nan
    return res


# ---------------------------------------------------------------- derivatives
def llsd(v: np.ndarray, rng_km: np.ndarray, daz_deg: float):
    """Azimuthal shear dV/ds and radial divergence dV/dr (s^-1) on a polar sweep (NaN = missing)."""
    naz, ng = v.shape
    ka = 2 if daz_deg < 0.75 else 1            # ~1 degree half-window in azimuth
    kg = 2                                     # +/- 2 gates (+/- 0.5 km)
    dr = float(rng_km[1] - rng_km[0])
    ok = np.isfinite(v)
    vz = np.where(ok, v, 0.0)
    pad = np.concatenate([vz[-ka:], vz, vz[:ka]], 0)
    pok = np.concatenate([ok[-ka:], ok, ok[:ka]], 0)
    num_a = np.zeros_like(vz); den_a = np.zeros_like(vz)
    num_r = np.zeros_like(vz); den_r = np.zeros_like(vz)
    cnt = np.zeros_like(vz)
    arc = rng_km[None, :] * np.radians(daz_deg)          # km per radial
    for da in range(-ka, ka + 1):
        for dg in range(-kg, kg + 1):
            sv = np.roll(pad[ka + da:ka + da + naz], -dg, axis=1)
            so = np.roll(pok[ka + da:ka + da + naz], -dg, axis=1)
            if dg > 0:
                so[:, -dg:] = False
            elif dg < 0:
                so[:, :-dg] = False
            ds = da * arc
            rr = dg * dr
            w = so.astype(np.float32)
            num_a += w * ds * sv; den_a += w * ds * ds
            num_r += w * rr * sv; den_r += w * rr * rr
            cnt += w
    # remove local mean to make the fit robust to missing gates
    mean = np.zeros_like(vz)
    for da in range(-ka, ka + 1):
        for dg in range(-kg, kg + 1):
            sv = np.roll(pad[ka + da:ka + da + naz], -dg, axis=1)
            so = np.roll(pok[ka + da:ka + da + naz], -dg, axis=1)
            mean += np.where(so, sv, 0)
    mean = mean / np.maximum(cnt, 1)
    sum_ds = np.zeros_like(vz); sum_rr = np.zeros_like(vz)
    for da in range(-ka, ka + 1):
        for dg in range(-kg, kg + 1):
            so = np.roll(pok[ka + da:ka + da + naz], -dg, axis=1)
            sum_ds += so * da * arc; sum_rr += so * dg * dr
    az_sh = (num_a - mean * sum_ds) / np.maximum(den_a, 1e-6) / 1000.0   # m/s per km -> s^-1
    div = (num_r - mean * sum_rr) / np.maximum(den_r, 1e-6) / 1000.0
    bad = ~ok | (cnt < 0.6 * (2 * ka + 1) * (2 * kg + 1))
    az_sh[bad] = np.nan; div[bad] = np.nan
    return az_sh.astype(np.float32), div.astype(np.float32)


# ---------------------------------------------------------------- volume
@dataclass
class Sweep:
    elev: float
    az: np.ndarray          # sorted azimuths (deg)
    rng: np.ndarray         # gate centers (km)
    data: dict              # field -> [naz, ngates] float32 (NaN missing), radials sorted by azimuth


def _get(radar, field, sl, order):
    if field not in radar.fields:
        return None
    a = radar.fields[field]["data"][sl]
    if np.ma.count(a) == 0:
        return None
    return np.ma.filled(a.astype(np.float32), np.nan)[order]


def read_volume(path: str | Path):
    import pyart
    radar = pyart.io.read_nexrad_archive(str(path))
    rng = radar.range["data"] / 1000.0
    by_elev: dict[float, Sweep] = {}
    nyq_of: dict[float, float] = {}
    for i in range(radar.nsweeps):
        sl = radar.get_slice(i)
        e = round(float(radar.fixed_angle["data"][i]), 1)
        az = radar.azimuth["data"][sl]
        order = np.argsort(az)
        sw = by_elev.setdefault(e, Sweep(e, az[order], rng, {}))
        if len(az) != len(sw.az):                      # keep one radial layout per elevation
            continue
        for f, name in (("Z", "reflectivity"), ("ZDR", "differential_reflectivity"),
                        ("CC", "cross_correlation_ratio"), ("V", "velocity"), ("SW", "spectrum_width")):
            if f in sw.data:
                continue
            if f == "Z" and _get(radar, "differential_reflectivity", sl, order) is None and \
                    _get(radar, "velocity", sl, order) is not None and any(
                        round(float(radar.fixed_angle["data"][j]), 1) == e and j != i for j in range(radar.nsweeps)):
                continue                                # split cut: take Z from the surveillance sweep
            a = _get(radar, name, sl, order)
            if a is not None:
                sw.data[f] = a
                if f == "V":
                    nyq_of[e] = float(radar.instrument_parameters["nyquist_velocity"]["data"][sl][0])
    sweeps = [by_elev[e] for e in sorted(by_elev)]
    t0 = radar.time["units"].split("since ")[-1]
    meta = {"site": radar.metadata.get("instrument_name", ""), "lat": float(radar.latitude["data"][0]),
            "lon": float(radar.longitude["data"][0]), "alt_m": float(radar.altitude["data"][0]),
            "vcp": int(radar.metadata.get("vcp_pattern", 0) or 0), "time_units": t0}
    return sweeps, nyq_of, meta


def layer_choice(elevs: list[float]):
    """For each layer: per grid cell, index of the sweep to use (-1 = none) and its slant range."""
    s, _ = grid_polar()
    H = np.stack([slant_and_height(s, e)[1] for e in elevs])     # [n_el, N, N]
    R = np.stack([slant_and_height(s, e)[0] for e in elevs])
    out = {}
    for L in set(LAYERS) | set(DP_LAYERS):
        if L == "low":
            idx = np.zeros(s.shape, int)
        else:
            d = np.abs(H - L)
            idx = np.argmin(d, axis=0)
            idx[np.take_along_axis(d, idx[None], 0)[0] > LAYER_TOL_KM] = -1
        idx[s > MAX_RANGE_KM] = -1
        r = np.take_along_axis(R, np.maximum(idx, 0)[None], 0)[0]
        out[L] = (idx, r)
    return out, H[0]


def sample(sw: Sweep, field: np.ndarray, sel: np.ndarray, r_km: np.ndarray, az_deg: np.ndarray, out: np.ndarray):
    if field is None or not sel.any():
        return
    a = sw.az
    k = np.searchsorted(a, az_deg[sel])
    k0 = np.clip(k - 1, 0, len(a) - 1); k1 = np.clip(k, 0, len(a) - 1)
    d0 = np.abs((az_deg[sel] - a[k0] + 180) % 360 - 180); d1 = np.abs((a[k1] - az_deg[sel] + 180) % 360 - 180)
    ai = np.where(d0 <= d1, k0, k1)
    # wrap: compare with first/last radial
    g = np.round((r_km[sel] - sw.rng[0]) / (sw.rng[1] - sw.rng[0])).astype(int)
    okg = (g >= 0) & (g < field.shape[1])
    vals = np.full(sel.sum(), np.nan, np.float32)
    vals[okg] = field[ai[okg], g[okg]]
    out[sel] = vals


NORM = {"Z": (0.0, 60.0, -0.5, 1.3), "AzSh": (0.0, 0.01, -3, 3), "Div": (0.0, 0.01, -3, 3),
        "SW": (0.0, 10.0, 0, 2), "ZDR": (0.0, 4.0, -1, 2), "CC": (0.8, 0.2, -2, 1)}


def process(path: str | Path, raw: bool = False):
    """Returns (x [NCH, N, N] float16 normalized, meta). With raw=True also the unnormalized layers."""
    sweeps, nyq_of, meta = read_volume(path)
    elevs = [sw.elev for sw in sweeps]
    choice, h_low = layer_choice(elevs)
    need = sorted({int(i) for L in LAYERS for i in np.unique(choice[L][0]) if i >= 0})
    vlist = [(k, sweeps[k]) for k in need if "V" in sweeps[k].data and sweeps[k].elev in nyq_of]
    dv = dealias([(sw.data["V"], nyq_of[sw.elev]) for _, sw in vlist])
    for (k, sw), v in zip(vlist, dv):
        daz = float(np.median(np.diff(sw.az)))
        sw.data["AzSh"], sw.data["Div"] = llsd(v, sw.rng, daz)
        sw.data["Vd"] = v
    s, az = grid_polar()
    x = np.zeros((NCH, N, N), np.float32)
    rawd = {}
    for ci, ch in enumerate(CHANNELS[:-2]):
        f, L = ch.split("_", 1)
        L = L if L == "low" else float(L)
        idx, r = choice[L]
        layer = np.full((N, N), np.nan, np.float32)
        for k in np.unique(idx):
            if k < 0:
                continue
            sample(sweeps[k], sweeps[k].data.get(f), idx == k, r, az, layer)
        if raw:
            rawd[ch] = layer.copy()
        m, sc, lo, hi = NORM[f]
        x[ci] = np.clip(np.nan_to_num((layer - m) / sc, nan=0.0), lo, hi)
    x[-2] = np.clip(s / MAX_RANGE_KM, 0, 1.2)
    x[-1] = np.clip(h_low / 5.0, 0, 2)
    meta["elevs"] = elevs
    return (x.astype(np.float16), meta, rawd) if raw else (x.astype(np.float16), meta)


if __name__ == "__main__":
    import time
    t = time.time()
    x, meta, rawd = process(sys.argv[1], raw=True)
    print(meta, round(time.time() - t, 1), "s")
    for ch in CHANNELS[:-2]:
        a = rawd[ch]
        print(f"{ch:10s} valid {np.isfinite(a).mean():.2f}  p1 {np.nanpercentile(a, 1):8.4f}  p99 {np.nanpercentile(a, 99):8.4f}")
