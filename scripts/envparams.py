"""Environmental parameters from one inflow sounding (MetPy + ECAPE helpers).

compute(z_agl, p, t, td, q, u, v, terrain, motion_obs) -> dict of the approved columns.
Units in: m AGL, hPa, degC, degC, kg/kg, m/s, m MSL, (u, v) m/s or None.
Storm-relative quantities are computed twice: _obs (observed motion) and _bunk (Bunkers RM).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import metpy.calc as mpcalc
from metpy.units import units

warnings.filterwarnings("ignore")
NAN = float("nan")
DZ = 10.0  # m, interpolation spacing for layer integrals


def _f(x, unit=None, nd=None):
    """pint/number -> float (rounded), NaN on failure."""
    try:
        if x is None:
            return NAN
        if hasattr(x, "to") and unit:
            x = x.to(unit)
        v = float(getattr(x, "magnitude", x))
        if not math.isfinite(v):
            return NAN
        return round(v, nd) if nd is not None else v
    except Exception:  # noqa: BLE001
        return NAN


class Sounding:
    def __init__(self, z, p, t, td, q, u, v, terrain):
        o = np.argsort(-p)                      # surface -> top (pressure decreasing)
        self.z, self.p, self.t, self.td, self.q = z[o], p[o], t[o], td[o], q[o]
        self.u, self.v, self.terrain = u[o], v[o], terrain
        self.P = self.p * units.hPa
        self.T = self.t * units.degC
        self.TD = self.td * units.degC
        self.lnp = np.log(self.p)

    # --- interpolation helpers
    def at_z(self, arr, h):
        return float(np.interp(h, self.z, arr, left=np.nan, right=np.nan))

    def z_at_p(self, pval):
        """height AGL at pressure (log-p interpolation)."""
        if pval is None or not np.isfinite(pval):
            return NAN
        return float(np.interp(-math.log(pval), -self.lnp, self.z, left=np.nan, right=np.nan))

    def grid(self, z0, z1):
        h = np.arange(z0, z1 + 1e-6, DZ)
        if h.size < 2 or h[-1] < z1:
            h = np.append(h, z1)
        return h, np.interp(h, self.z, self.u), np.interp(h, self.z, self.v)

    def wind(self, h):
        return self.at_z(self.u, h), self.at_z(self.v, h)


# ----------------------------------------------------------------------------- parcels
def _parcel_stats(s: Sounding, k0: int, t0, td0):
    """CAPE/CIN/LFC/EL for a parcel starting at index k0 with (t0, td0)."""
    P, T, TD = s.P[k0:], s.T[k0:], s.TD[k0:]
    prof = mpcalc.parcel_profile(P, t0, td0)
    cape, cin = mpcalc.cape_cin(P, T, TD, prof)
    try:
        lfc_p, _ = mpcalc.lfc(P, T, TD, prof)
    except Exception:  # noqa: BLE001
        lfc_p = None
    try:
        el_p, _ = mpcalc.el(P, T, TD, prof)
    except Exception:  # noqa: BLE001
        el_p = None
    lcl_p, _ = mpcalc.lcl(P[0], t0, td0)
    return {
        "cape": _f(cape, "J/kg"), "cin": _f(cin, "J/kg"),
        "lcl": s.z_at_p(_f(lcl_p, "hPa")), "lfc": s.z_at_p(_f(lfc_p, "hPa")), "el": s.z_at_p(_f(el_p, "hPa")),
        "prof": prof,
    }


def _cape_below(s: Sounding, k0, t0, td0, ztop):
    """CAPE of a parcel integrated only up to ztop (m AGL)."""
    ptop = math.exp(np.interp(ztop, s.z, s.lnp))
    keep = s.p >= ptop
    keep[:k0] = False
    idx = np.where(keep)[0]
    if idx.size < 3:
        return NAN
    P, T, TD = s.P[idx], s.T[idx], s.TD[idx]
    prof = mpcalc.parcel_profile(P, t0, td0)
    cape, _ = mpcalc.cape_cin(P, T, TD, prof)
    return _f(cape, "J/kg")


def _ecape(s: Sounding, sr01: float, cape_ml: float):
    """ECAPE (Peters et al.) for the mixed-layer parcel, with a given 0-1 km mean SR wind speed."""
    from ecape import calc as ec
    if not np.isfinite(sr01) or not np.isfinite(cape_ml) or cape_ml <= 0 or sr01 <= 0:
        return 0.0 if cape_ml == 0 else NAN
    # Uniform 50 m grid: the package's mean MSE is an unweighted level average, which equals
    # Peters et al.'s height average (surface to z) only on evenly spaced levels.
    zz = np.arange(s.z[0], s.z[-1], 50.0)
    P = np.exp(np.interp(zz, s.z, s.lnp)) * units.hPa
    T = np.interp(zz, s.z, s.t) * units.degC
    q = np.interp(zz, s.z, s.q) * units("kg/kg")
    TD = mpcalc.dewpoint_from_specific_humidity(P, q)
    zmsl = (zz + s.terrain) * units.m
    lfc_idx, _ = ec.calc_lfc_height(P, zmsl, T, TD, mpcalc.mixed_parcel)
    el_idx, el_z = ec.calc_el_height(P, zmsl, T, TD, mpcalc.mixed_parcel)
    mse_bar, mse_star = ec.calc_mse(P, zmsl, T, q)
    arg = ec.calc_integral_arg(mse_bar, mse_star, T)
    ncape = ec.calc_ncape(arg, zmsl, lfc_idx, el_idx)
    psi = ec.calc_psi(el_z)
    e = ec.calc_ecape_a(sr01 * units("m/s"), psi, ncape, cape_ml * units("J/kg"))
    return _f(e, "J/kg")


def _effective_layer(s: Sounding):
    """Thompson et al. (2007): lowest contiguous layer with CAPE >= 100 and CIN >= -250 J/kg."""
    base = top = None
    for k in range(len(s.p)):
        if s.p[0] - s.p[k] > 300:
            break
        try:
            prof = mpcalc.parcel_profile(s.P[k:], s.T[k], s.TD[k])
            cape, cin = mpcalc.cape_cin(s.P[k:], s.T[k:], s.TD[k:], prof)
            ok = _f(cape) >= 100 and _f(cin) >= -250
        except Exception:  # noqa: BLE001
            ok = False
        if ok and base is None:
            base = k
        if base is not None:
            if ok:
                top = k
            else:
                break
    if base is None:
        return NAN, NAN
    return float(s.z[base]), float(s.z[top])


# ----------------------------------------------------------------------------- kinematics
def _bwd(s, z0, z1):
    u0, v0 = s.wind(z0)
    u1, v1 = s.wind(z1)
    return math.hypot(u1 - u0, v1 - v0)


def _srh(s, z0, z1, cu, cv):
    if not (np.isfinite(z0) and np.isfinite(z1)) or z1 <= z0:
        return 0.0
    _, u, v = s.grid(z0, z1)
    ur, vr = u - cu, v - cv
    return float(np.sum(ur[1:] * vr[:-1] - ur[:-1] * vr[1:]))


def _mean_srf(s, z0, z1, cu, cv):
    _, u, v = s.grid(z0, z1)
    return float(np.mean(np.hypot(u - cu, v - cv)))


def _vort500(s, cu, cv):
    h, u, v = s.grid(s.z[0], 500.0)
    dudz, dvdz = np.gradient(u, h), np.gradient(v, h)
    wx, wy = -dvdz, dudz                       # horizontal vorticity
    ur, vr = u - cu, v - cv
    sp = np.hypot(ur, vr)
    sp[sp == 0] = np.nan
    sx, sy = ur / sp, vr / sp
    sw = wx * sx + wy * sy                     # streamwise (signed)
    cw = np.abs(wx * sy - wy * sx)             # crosswise magnitude
    mag = np.hypot(wx, wy)
    sw_m, cw_m, mag_m = np.nanmean(sw), np.nanmean(cw), np.nanmean(mag)
    psw = 100 * sw_m / mag_m if mag_m > 0 else NAN
    return float(sw_m), float(cw_m), float(psw)


def _crit_angle(s, cu, cv):
    u0, v0 = s.u[0], s.v[0]
    u5, v5 = s.wind(500.0)
    a = np.array([u5 - u0, v5 - v0])          # 0-500 m shear
    b = np.array([u0 - cu, v0 - cv])          # lowest-level storm-relative inflow
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return NAN
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1, 1))))


# ----------------------------------------------------------------------------- composites
def _clip(x, lo, hi):
    return min(max(x, lo), hi)


def stp_fixed(sbcape, sblcl, srh1, bwd6):
    if not all(np.isfinite([sbcape, sblcl, srh1, bwd6])):
        return NAN
    lcl = 1.0 if sblcl < 1000 else 0.0 if sblcl > 2000 else (2000 - sblcl) / 1000
    shr = 0.0 if bwd6 < 12.5 else min(bwd6, 30) / 20
    return (sbcape / 1500) * lcl * (srh1 / 150) * shr


def stp_eff(mlcape, mllcl, srh, srh_norm, ebwd, mlcin):
    if not all(np.isfinite([mlcape, mllcl, srh, ebwd, mlcin])):
        return NAN
    lcl = 1.0 if mllcl < 1000 else 0.0 if mllcl > 2000 else (2000 - mllcl) / 1000
    shr = 0.0 if ebwd < 12.5 else min(ebwd, 30) / 20
    cin = 1.0 if mlcin > -50 else 0.0 if mlcin < -200 else (200 + mlcin) / 150
    return (mlcape / 1500) * lcl * (srh / srh_norm) * shr * cin


def scp(mucape, esrh, ebwd, mucin):
    if not all(np.isfinite([mucape, esrh, ebwd, mucin])):
        return NAN
    shr = 0.0 if ebwd < 10 else 1.0 if ebwd > 20 else ebwd / 20
    cin = 1.0 if mucin > -40 else -40 / mucin
    return (mucape / 1000) * (esrh / 50) * shr * cin


# ----------------------------------------------------------------------------- main entry
def compute(z, p, t, td, q, u, v, terrain, motion_obs):
    s = Sounding(np.asarray(z, float), np.asarray(p, float), np.asarray(t, float), np.asarray(td, float),
                 np.asarray(q, float), np.asarray(u, float), np.asarray(v, float), float(terrain))
    out = {}

    def grab(fn, *names):
        try:
            vals = fn()
            vals = vals if isinstance(vals, tuple) else (vals,)
            for n, val in zip(names, vals):
                out[n] = val
        except Exception:  # noqa: BLE001
            for n in names:
                out[n] = NAN

    # storm motions
    def motions():
        rm, lm, mw = mpcalc.bunkers_storm_motion(s.P, s.u * units("m/s"), s.v * units("m/s"), s.z * units.m)
        return (_f(rm[0], "m/s"), _f(rm[1], "m/s"), _f(lm[0], "m/s"), _f(lm[1], "m/s"), _f(mw[0], "m/s"), _f(mw[1], "m/s"))
    grab(motions, "bunk_rm_u", "bunk_rm_v", "bunk_lm_u", "bunk_lm_v", "mw06_u", "mw06_v")

    # parcels
    def sb():
        r = _parcel_stats(s, 0, s.T[0], s.TD[0])
        return r["cape"], r["cin"], r["lcl"], r["lfc"], r["el"]
    grab(sb, "sbcape", "sbcin", "sblcl", "sblfc", "sbel")

    ml_t0 = ml_td0 = None
    def ml():
        nonlocal ml_t0, ml_td0
        _, ml_t0, ml_td0 = mpcalc.mixed_parcel(s.P, s.T, s.TD, depth=100 * units.hPa)
        cape, cin = mpcalc.mixed_layer_cape_cin(s.P, s.T, s.TD, depth=100 * units.hPa)
        r = _parcel_stats(s, 0, ml_t0, ml_td0)
        return _f(cape, "J/kg"), _f(cin, "J/kg"), r["lcl"], r["lfc"], r["el"]
    grab(ml, "mlcape", "mlcin", "mllcl", "mllfc", "mlel")

    def mu():
        pp, t0, td0, k = mpcalc.most_unstable_parcel(s.P, s.T, s.TD, depth=300 * units.hPa)
        cape, cin = mpcalc.most_unstable_cape_cin(s.P, s.T, s.TD, depth=300 * units.hPa)
        r = _parcel_stats(s, int(k), t0, td0)
        return _f(cape, "J/kg"), _f(cin, "J/kg"), float(s.z[int(k)]), r["lfc"], r["el"]
    grab(mu, "mucape", "mucin", "mulpl", "mulfc", "muel")

    grab(lambda: _cape_below(s, 0, ml_t0, ml_td0, 3000.0) if ml_t0 is not None else NAN, "mlcape03")
    grab(lambda: _effective_layer(s), "eff_base", "eff_top")
    out["eff_depth"] = out["eff_top"] - out["eff_base"] if np.isfinite(out.get("eff_base", NAN)) else NAN
    grab(lambda: _f(mpcalc.downdraft_cape(s.P, s.T, s.TD)[0], "J/kg"), "dcape")

    # lapse rates & moisture
    grab(lambda: (s.at_z(s.t, 0 + s.z[0]) - s.at_z(s.t, 3000)) / 3.0, "lr03")
    grab(lambda: (s.at_z(s.t, 3000) - s.at_z(s.t, 6000)) / 3.0, "lr36")
    def lr75():
        z7, z5 = s.z_at_p(700), s.z_at_p(500)
        return (s.at_z(s.t, z7) - s.at_z(s.t, z5)) / ((z5 - z7) / 1000)
    grab(lr75, "lr75")
    out["t_sfc"], out["td_sfc"] = float(s.t[0]), float(s.td[0])
    def mlmr():
        sel = s.p >= s.p[0] - 100
        return float(np.mean(s.q[sel] / (1 - s.q[sel])) * 1000)
    grab(mlmr, "mlmr")
    grab(lambda: _f(mpcalc.precipitable_water(s.P, s.TD), "mm"), "pw")
    def rh():
        r = np.asarray(mpcalc.relative_humidity_from_dewpoint(s.T, s.TD).to("percent").magnitude)
        m = lambda a, b: float(np.mean(np.interp(np.arange(a, b + 1, DZ), s.z, r)))
        return m(s.z[0], 1000), m(1000, 3000), m(3000, 6000)
    grab(rh, "rh01", "rh13", "rh36")
    def frz():
        if s.t[0] <= 0:
            return 0.0
        k = int(np.argmax(s.t <= 0))
        return float(np.interp(0, [s.t[k], s.t[k - 1]], [s.z[k], s.z[k - 1]]))
    grab(frz, "frz_lvl")

    # winds & shear (motion-independent)
    sp = lambda h: math.hypot(*s.wind(h))
    grab(lambda: (sp(500), sp(1000), sp(6000)), "grw500", "grw1", "grw6")
    def grw05_3():
        _, u, v = s.grid(500, 3000)
        return float(np.mean(np.hypot(u, v)))
    grab(grw05_3, "grw05_3")
    z0 = float(s.z[0])
    grab(lambda: (_bwd(s, z0, 500), _bwd(s, z0, 1000), _bwd(s, z0, 3000), _bwd(s, z0, 6000), _bwd(s, 1000, 3000)),
         "bwd500", "bwd1", "bwd3", "bwd6", "bwd13")
    def ebwd():
        b, el = out.get("eff_base", NAN), out.get("muel", NAN)
        if not np.isfinite(b) or not np.isfinite(el):
            return 0.0
        return _bwd(s, b, b + 0.5 * (el - b))
    grab(ebwd, "ebwd")

    # storm-relative, twice
    motions_sr = {"obs": motion_obs, "bunk": (out.get("bunk_rm_u"), out.get("bunk_rm_v"))}
    for tag, mot in motions_sr.items():
        names = ["srh500", "srh1", "srh3", "srh13", "esrh", "srf500", "srf1", "srf3", "srf13", "srf01",
                 "sw500", "cw500", "psw500", "crit_angle", "ecape", "stp_fixed", "stp_eff", "stp500", "scp"]
        if mot is None or not all(np.isfinite(mot)):
            for n in names:
                out[f"{n}_{tag}"] = NAN
            continue
        cu, cv = mot
        grab(lambda: (_srh(s, z0, 500, cu, cv), _srh(s, z0, 1000, cu, cv), _srh(s, z0, 3000, cu, cv),
                      _srh(s, 1000, 3000, cu, cv)), f"srh500_{tag}", f"srh1_{tag}", f"srh3_{tag}", f"srh13_{tag}")
        grab(lambda: _srh(s, out["eff_base"], out["eff_top"], cu, cv) if np.isfinite(out.get("eff_base", NAN)) else 0.0, f"esrh_{tag}")
        grab(lambda: tuple(math.hypot(s.wind(h)[0] - cu, s.wind(h)[1] - cv) for h in (500, 1000, 3000)),
             f"srf500_{tag}", f"srf1_{tag}", f"srf3_{tag}")
        grab(lambda: (_mean_srf(s, 1000, 3000, cu, cv), _mean_srf(s, z0, 1000, cu, cv)), f"srf13_{tag}", f"srf01_{tag}")
        grab(lambda: _vort500(s, cu, cv), f"sw500_{tag}", f"cw500_{tag}", f"psw500_{tag}")
        grab(lambda: _crit_angle(s, cu, cv), f"crit_angle_{tag}")
        grab(lambda: _ecape(s, out[f"srf01_{tag}"], out.get("mlcape", NAN)), f"ecape_{tag}")
        out[f"stp_fixed_{tag}"] = stp_fixed(out.get("sbcape", NAN), out.get("sblcl", NAN), out[f"srh1_{tag}"], out.get("bwd6", NAN))
        out[f"stp_eff_{tag}"] = stp_eff(out.get("mlcape", NAN), out.get("mllcl", NAN), out[f"esrh_{tag}"], 150, out.get("ebwd", NAN), out.get("mlcin", NAN))
        out[f"stp500_{tag}"] = stp_eff(out.get("mlcape", NAN), out.get("mllcl", NAN), out[f"srh500_{tag}"], 75, out.get("ebwd", NAN), out.get("mlcin", NAN))
        out[f"scp_{tag}"] = scp(out.get("mucape", NAN), out[f"esrh_{tag}"], out.get("ebwd", NAN), out.get("mucin", NAN))

    # filters from Coffer et al.
    mucape = out.get("mucape", NAN)
    out["flag_mucape0"] = "yes" if (np.isfinite(mucape) and mucape <= 0) else ""
    eb = out.get("eff_base", NAN)
    out["flag_eib_elev"] = "yes" if (not np.isfinite(eb) or eb > s.z[0] + 1) else ""
    out["flag_ebwd_lt12p5"] = "yes" if out.get("ebwd", 0) < 12.5 else ""
    return out
